from huggingface_hub import (file_exists, hf_hub_download,
                             try_to_load_from_cache, list_repo_files)
import os
HF_TOKEN = os.getenv('HF_TOKEN', None)

a = file_exists("facebook/bart-base", "sentence_albert_config.json", token=HF_TOKEN)
print(a)

b = list_repo_files("facebook/bart-base", token=HF_TOKEN)
print(b)
