import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
import json
from typing import Union, OrderedDict
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

@router.post("/gen_by_lora")
def gen_by_lora(lora_name:str, prompts:list[str]):
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
    saved_path = f"/root/autodl-tmp/aitoolkit/gen/{lora_name}"
    
    config['config']['name'] = lora_name
    config['config']['process'][0]['output_folder'] = saved_path
    config['config']['process'][0]['generate']['prompts'] = prompts
    config['config']['process'][0]['model']['lora_path'] = f"/root/ai-toolkit/output/{lora_name}/{lora_name}.safetensors"

    os.makedirs(saved_path, exist_ok=True)
    with open(f"{saved_path}/{lora_name}.yaml", 'w') as f:
        json.dump(config, f, indent=4)
    config_file = f"{saved_path}/{lora_name}.yaml"


    try:
        job = get_job(config_file, args.name)
        job.run()
        # job.cleanup()
        jobs_completed += 1
    except Exception as e:
        print(f"Error running job: {e}")
        jobs_failed += 1
        if not args.recover:
            print_end_message(jobs_completed, jobs_failed)
            raise e

    
if __name__ == '__main__':
    pass
