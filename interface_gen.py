import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import json
from typing import Union, OrderedDict
from pydantic import BaseModel
from routers import ImageResponse, logger
from routers.obs_client import obs_upload_file
from fastapi import APIRouter
from dotenv import load_dotenv
from toolkit.config import get_config
# Load the .env file if it exists
load_dotenv()

sys.path.insert(0, os.getcwd())
# must come before ANY torch or fastai imports
# import toolkit.cuda_malloc

# turn off diffusers telemetry until I can figure out how to make it opt-in
os.environ['DISABLE_TELEMETRY'] = 'YES'

# check if we have DEBUG_TOOLKIT in env
if os.environ.get("DEBUG_TOOLKIT", "0") == "1":
    # set torch to trace mode
    import torch
    torch.autograd.set_detect_anomaly(True)
import argparse
from toolkit.job import get_job

lora_path = "/root/autodl-tmp/aitoolkit/dataset/loras"
imgsave_path = "/root/autodl-tmp/aitoolkit/gen"

def print_end_message(jobs_completed, jobs_failed):
    failure_string = f"{jobs_failed} failure{'' if jobs_failed == 1 else 's'}" if jobs_failed > 0 else ""
    completed_string = f"{jobs_completed} completed job{'' if jobs_completed == 1 else 's'}"

    print("")
    print("========================================")
    print("Result:")
    if len(completed_string) > 0:
        print(f" - {completed_string}")
    if len(failure_string) > 0:
        print(f" - {failure_string}")
    print("========================================")


router = APIRouter(
    prefix="/gen_by_lora",
    tags=["model"],
    dependencies=[],
    responses={404: {"description": "Not found"}},
)

class GenByLoraRequest(BaseModel):
    lora_name: str
    prompts: list[str]

@router.post("/gen_by_lora")
def gen_by_lora(request: GenByLoraRequest):
    try:
        lora_name = request.lora_name
        prompts = request.prompts
        class Args():
            def __init__(self):
                self.config_file = 'config/generate.yaml'
                self.recover = False
                self.name = None

        args = Args()

        jobs_completed = 0
        jobs_failed = 0

        config_file = args.config_file
        config = get_config(config_file, args.name)
        
        config['config']['name'] = lora_name
        config['config']['process'][0]['output_folder'] = f"{imgsave_path}/{lora_name}"
        config['config']['process'][0]['generate']['prompts'] = prompts
        config['config']['process'][0]['model']['lora_path'] = f"{lora_path}/{lora_name}/{lora_name}.safetensors"

        os.makedirs(f"{imgsave_path}", exist_ok=True)
        config_file = f"{imgsave_path}/{lora_name}.yaml"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error while processing the config file: {e}")
        raise Exception(f"Error while processing the config file: {e}")
    try:
        job = get_job(config_file, args.name)
        job.run()
        job.cleanup()
        jobs_completed += 1

        all_images = os.listdir(f"{imgsave_path}/{lora_name}")
        len_tmp = len(prompts)
        generated_images = all_images[-len_tmp:]
        image_urls = []
        for img in generated_images:
            img_path = f"{imgsave_path}/{lora_name}/{img}"
            img_url = obs_upload_file(img_path, f"aitoolkit/gen_by_lora/{lora_name}")
            image_urls.append(img_url)
        return ImageResponse(code=200, message="Success", data=image_urls)
        
    except Exception as e:
        logger.error(f"Error running job: {e}")
        jobs_failed += 1
        if not args.recover:
            print_end_message(jobs_completed, jobs_failed)
            raise Exception(f"Error running job: {e}")    
        return ImageResponse(code=500, message=f"Failed;{e}", data=[])
        

    
if __name__ == '__main__':
    pass
