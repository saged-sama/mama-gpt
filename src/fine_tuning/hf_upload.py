from huggingface_hub import HfApi, login
from dotenv import load_dotenv
import os

load_dotenv("./.env")

login(os.getenv("HF_TOKEN") or "")

api = HfApi()
commit_info = api.upload_folder(
    folder_path="./out/lora/qlora_llama_3",
    repo_id="sagedsama/lora_practice",
    repo_type="model",
    commit_message="This is just my first lora practice and test",
    commit_description="",
)

print(commit_info.commit_url)