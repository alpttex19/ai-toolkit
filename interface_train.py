import os, logging
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import whoami    
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys

# Add the current working directory to the Python path
sys.path.insert(0, os.getcwd())

from routers import logger, SAVE_PATH
from routers.obs_client import obs_upload_file
from fastapi import APIRouter, File, UploadFile
from datetime import datetime
import zipfile
from PIL import Image
import torch
import uuid
import os
import shutil
import json
import yaml
from slugify import slugify
from transformers import AutoProcessor, AutoModelForCausalLM

sys.path.insert(0, "ai-toolkit")
from toolkit.job import get_job

MAX_IMAGES = 150
saved_path = f"{SAVE_PATH}/dataset"

def create_dataset(imagepaths, captions, saved_path, lora_name):
    print("Creating dataset")
    images = imagepaths
    time_suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
    destination_folder = str(f"{saved_path}/{lora_name}/{time_suffix}")
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    jsonl_file_path = os.path.join(destination_folder, "metadata.jsonl")
    with open(jsonl_file_path, "a") as jsonl_file:
        for index, image in enumerate(images):
            new_image_path = shutil.copy(image, destination_folder)

            original_caption = captions[index]
            file_name = os.path.basename(new_image_path)

            data = {"file_name": file_name, "prompt": original_caption}

            jsonl_file.write(json.dumps(data) + "\n")

    return destination_folder


def run_captioning(images, imageurls, captions):
    #Load internally to not consume resources for training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        "multimodalart/Florence-2-large-no-flash-attn", torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained("multimodalart/Florence-2-large-no-flash-attn", trust_remote_code=True)

    img_caption_list = []
    captions = list(captions)
    for i, image_path in enumerate(images):
        print(captions[i])
        if isinstance(image_path, str):  # If image is a file path
            image = Image.open(image_path).convert("RGB")

        prompt = "<DETAILED_CAPTION>"
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)

        generated_ids = model.generate(
            input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=1024, num_beams=3
        )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = processor.post_process_generation(
            generated_text, task=prompt, image_size=(image.width, image.height)
        )
        caption_text = parsed_answer["<DETAILED_CAPTION>"].replace("The image shows ", "")
        caption_text = f"{caption_text} [trigger]"
        captions[i] = caption_text
        img_caption_list.append(
            {
                "name": os.path.basename(image_path), 
                "caption": caption_text, 
                "url": imageurls[i]
            }
        )

    model.to("cpu")
    del model
    del processor
    return img_caption_list

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and v:
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def start_training(
    saved_path,
    lora_name,
    concept_sentence,
    steps,
    lr,
    rank,
    model_to_train,
    low_vram,
    dataset_folder,
    sample_1,
    sample_2,
    sample_3,
    use_more_advanced_options,
    more_advanced_options,
):
    push_to_hub = False
    if not lora_name:
        logger.error("You forgot to insert your LoRA name! This name has to be unique.")
                
    print("Started training")
    slugged_lora_name = slugify(lora_name)

    # Load the default config
    with open("config/my_first_lora.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Update the config with user inputs
    config["config"]["name"] = slugged_lora_name
    config["config"]["process"][0]["model"]["low_vram"] = low_vram
    config["config"]["process"][0]["training_folder"] = saved_path
    config["config"]["process"][0]["train"]["skip_first_sample"] = True
    config["config"]["process"][0]["train"]["steps"] = int(steps)
    config["config"]["process"][0]["train"]["lr"] = float(lr)
    config["config"]["process"][0]["network"]["linear"] = int(rank)
    config["config"]["process"][0]["network"]["linear_alpha"] = int(rank)
    config["config"]["process"][0]["datasets"][0]["folder_path"] = dataset_folder
    config["config"]["process"][0]["save"]["push_to_hub"] = push_to_hub
    if(push_to_hub):
        try:
            username = whoami()["name"]
        except:
            logger.error("Error trying to retrieve your username. Are you sure you are logged in with Hugging Face?")
        
        config["config"]["process"][0]["save"]["hf_repo_id"] = f"{username}/{slugged_lora_name}"
        config["config"]["process"][0]["save"]["hf_private"] = True
    if concept_sentence:
        config["config"]["process"][0]["trigger_word"] = concept_sentence
    
    if sample_1 or sample_2 or sample_3:
        config["config"]["process"][0]["train"]["disable_sampling"] = False
        config["config"]["process"][0]["sample"]["sample_every"] = steps
        config["config"]["process"][0]["sample"]["sample_steps"] = 20
        config["config"]["process"][0]["sample"]["prompts"] = []
        if sample_1:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_1)
        if sample_2:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_2)
        if sample_3:
            config["config"]["process"][0]["sample"]["prompts"].append(sample_3)
    else:
        config["config"]["process"][0]["train"]["disable_sampling"] = True
    if(model_to_train == "schnell"):
        config["config"]["process"][0]["model"]["name_or_path"] = "black-forest-labs/FLUX.1-schnell"
        config["config"]["process"][0]["model"]["assistant_lora_path"] = "ostris/FLUX.1-schnell-training-adapter"
        config["config"]["process"][0]["sample"]["sample_steps"] = 4
    if(use_more_advanced_options):
        more_advanced_options_dict = yaml.safe_load(more_advanced_options)
        config["config"]["process"][0] = recursive_update(config["config"]["process"][0], more_advanced_options_dict)
        print(config)
    
    # Save the updated config
    # generate a random name for the config
    config_path = f"{saved_path}/{lora_name}/{slugged_lora_name}.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # run the job locally
    job = get_job(config_path)
    job.run()
    job.cleanup()

    return f"Training completed successfully. Model saved as {slugged_lora_name}"

router = APIRouter(
    prefix="/train_lora",
    tags=["model"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

@router.post("/custom_captioning")
def custom_captioning(
                                files_zip:UploadFile = File(...), 
                                auto_caption:bool = False
                            ):
    try:
        #Extract images
        unique_id = uuid.uuid4()
        image_path = f"{saved_path}/images-{unique_id}"
        os.makedirs(image_path, exist_ok=True)
        with open("images.zip", "wb") as buffer:
            shutil.copyfileobj(files_zip.file, buffer)

        shutil.rmtree(image_path, ignore_errors=True)
        with zipfile.ZipFile("images.zip", "r") as zip_ref:
            zip_ref.extractall(path=image_path)
            os.remove("images.zip")
        
        caption_cont_list = []
        imagepath_list = []
        imageurl_list = []
        for file in os.listdir(image_path):
            if not file.endswith(".txt"):
                imagepath_list.append(f"{image_path}/{file}")
                imageurl = obs_upload_file(file_path=f"{image_path}/{file}", dir=f"aitoolkit/images-{unique_id}")
                imageurl_list.append(imageurl)
                # 如果是自动标注，就不需要caption
                if auto_caption:
                    caption_cont_list.append("[trigger]")
                # 如果是手动标注，就需要读取caption
                else:
                    base_name = os.path.splitext(file)[0]
                    # 如果没有caption文件，就用[trigger]
                    if not os.path.exists(f"{image_path}/{base_name}.txt"):
                        caption_cont_list.append("[trigger]")
                    else:
                        with open(f"{image_path}/{base_name}.txt", "r") as file:
                            caption_cont_list.append(file.read() + " [trigger]")
        if auto_caption:
            img_caption_list = run_captioning(imagepath_list, imageurl_list, caption_cont_list)
        else:
            img_caption_list = []
            for i, image_path in enumerate(imagepath_list):
                img_caption_list.append(
                    {
                        "name": os.path.basename(image_path), 
                        "caption": caption_cont_list[i], 
                        "url": imageurl_list[i]
                    }
                )
        return {"unique_id": unique_id, "img_caption_list": img_caption_list}
        
    except Exception as e:
        logger.error(f"Error in custom_captioning: {e}")
        raise Exception("Error in custom_captioning")



@router.post("/train_lora")
def train_lora(   
                        lora_name:str, 
                        concept_sentence:str,
                        imgcaption_and_id:dict,
                        steps:int = 1000,
                        lr:float = 4e-4,
                        rank:int = 16,
                        model_to_train:str = "dev",
                        low_vram:bool = True,
                    ):
    try:
        lora_unique_id = uuid.uuid4()
        lora_name = f"{lora_name}-{lora_unique_id}"
        unique_id = imgcaption_and_id["unique_id"]
        img_caption_list = imgcaption_and_id["img_caption_list"]
        imagepath_list = []
        caption_cont_list = []
        for dict in img_caption_list:
            imagepath_list.append(f"{saved_path}/images-{unique_id}/{dict['name']}")
            caption_cont_list.append(dict["caption"])

        dataset_folder = create_dataset(imagepath_list, caption_cont_list, saved_path, lora_name)
        
        # 配置查询状态 logger
        process_query = logging.getLogger('process_query')
        process_query.setLevel(logging.INFO)
        # 配置处理程序
        file_handler2 = logging.FileHandler(f'{saved_path}/train_log.log')
        formatter2 = logging.Formatter('%(name)s: %(message)s')
        file_handler2.setFormatter(formatter2)
        process_query.addHandler(file_handler2)

        class StreamToLogger:
            def __init__(self, level=logging.INFO):
                self.level = level

            def write(self, message):
                # 仅处理非空消息
                if message.rstrip():  
                    process_query.info(message.rstrip())

            def flush(self):
                pass
        # 保存原始sys.stdout，以便恢复
        original_stdout = sys.stdout

        # 将sys.stdout重定向到StreamToLogger对象
        sys.stdout = StreamToLogger()
        
        progress_area = start_training(
            saved_path,
            lora_name,
            concept_sentence,
            steps,
            lr,
            rank,
            model_to_train,
            low_vram,
            dataset_folder,
            None,
            None,
            None,
            False,
            None
        )
        # 恢复标准输出
        sys.stdout = original_stdout

        return lora_name
    
    except Exception as e:
        logger.error(f"Error in train_lora: {e}")
        raise Exception("Error in train_lora")
    
@router.post("/process_query")
def process_query():
    try:
        with open(f"{saved_path}/train_log.log", "r") as f:
            process_query = f.read()
            last_msg = process_query.split("process_query:")[-1]
        return {"last_msg": last_msg}
    except Exception as e:
        logger.error(f"Error in process_query: {e}")
        raise Exception("Error in process_query")

if __name__ == "__main__":
    pass

config_yaml = '''
device: cuda:0
model:
  is_flux: true
  quantize: true
network:
  linear: 16 #it will overcome the 'rank' parameter
  linear_alpha: 16 #you can have an alpha different than the ranking if you'd like
  type: lora
sample:
  guidance_scale: 3.5
  height: 1024
  neg: '' #doesn't work for FLUX
  sample_every: 1000
  sample_steps: 28
  sampler: flowmatch
  seed: 42
  walk_seed: true
  width: 1024
save:
  dtype: float16
  hf_private: true
  max_step_saves_to_keep: 4
  push_to_hub: true
  save_every: 10000
train:
  batch_size: 1
  dtype: bf16
  ema_config:
    ema_decay: 0.99
    use_ema: true
  gradient_accumulation_steps: 1
  gradient_checkpointing: true
  noise_scheduler: flowmatch 
  optimizer: adamw8bit #options: prodigy, dadaptation, adamw, adamw8bit, lion, lion8bit
  train_text_encoder: false #probably doesn't work for flux
  train_unet: true
'''

