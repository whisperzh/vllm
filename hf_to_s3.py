# SPDX-License-Identifier: Apache-2.0
import logging
import os
import shutil

import boto3
from huggingface_hub import HfApi, snapshot_download
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTransfer:

    def __init__(self,
                 model_id,
                 s3_bucket,
                 aws_access_key_id=None,
                 aws_secret_access_key=None,
                 aws_region=None):
        """
        Initialize the ModelTransfer class.
        
        Args:
            model_id (str): HuggingFace model ID 
            s3_bucket (str): Name of the S3 bucket
            aws_access_key_id (str, optional)
            aws_secret_access_key (str, optional)
            aws_region (str, optional): AWS region. Defaults to None.
        """
        self.model_id = model_id
        self.s3_bucket = s3_bucket
        self.model_name = model_id.split('/')[-1]

        # Initialize S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region)

        # Initialize Hugging Face API
        self.hf_api = HfApi()

    def download_model(self, local_dir):
        """
        Download the model from HuggingFace.
        
        Args:
            local_dir (str): Local directory to save the model
        
        Returns:
            str: Path to the downloaded model directory
        """
        logger.info("Downloading model %s...", self.model_id)

        try:
            local_dir_with_model = os.path.join(local_dir, self.model_name)
            snapshot_download(repo_id=self.model_id,
                              local_dir=local_dir_with_model,
                              local_dir_use_symlinks=False,
                              token=os.getenv("HF_TOKEN"))
            logger.info("Model downloaded successfully to %s",
                        local_dir_with_model)
            return local_dir_with_model

        except Exception as e:
            logger.error("Error downloading model: %s", str(e))
            raise

    def upload_to_s3(self, local_dir):
        """
        Upload the model directory to S3.
        
        Args:
            local_dir (str): Local directory containing the model files
        """
        logger.info("Uploading model to S3 bucket %s...", self.s3_bucket)

        try:
            # Walk through all files in the directory
            for root, _, files in os.walk(local_dir):
                for filename in files:
                    # Get the full local path
                    local_path = os.path.join(root, filename)

                    # Calculate S3 path (preserve directory structure)
                    relative_path = os.path.relpath(local_path, local_dir)
                    s3_path = f"{self.model_name}/{relative_path}"

                    # Upload file with progress bar
                    file_size = os.path.getsize(local_path)
                    with tqdm(total=file_size,
                              unit='B',
                              unit_scale=True,
                              desc=f"Uploading {filename}") as pbar:
                        self.s3_client.upload_file(
                            local_path,
                            self.s3_bucket,
                            s3_path,
                            Callback=lambda bytes_transferred: pbar.update(
                                bytes_transferred))

                    logger.info("Uploaded %s to s3://%s/%s", filename,
                                self.s3_bucket, s3_path)

            logger.info("Model upload completed successfully!")

        except Exception as e:
            logger.error("Error uploading to S3: %s", str(e))
            raise


def main():
    # Configuration
    MODEL_ID = [
        "facebook/opt-350m",
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2-1.5B-Instruct",
        "meta-llama/Llama-3.2-1B-Instruct",
        "microsoft/Phi-3.5-vision-instruct",
        "HuggingFaceH4/zephyr-7b-beta",
        "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
        "ai21labs/Jamba-tiny-dev",
        "google/gemma-2-2b-it",
        "google/gemma-2b",
        "google/gemma-2-9b",
        "google/gemma-7b",
    ]
    S3_BUCKET = "vllm-ci-model-weights"  # Replace with your S3 bucket name
    LOCAL_DIR = "~/models"  # Local directory to temporarily store the model

    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = "us-west-2"

    # Create transfer object
    for model_id in MODEL_ID:
        transfer = ModelTransfer(model_id=model_id,
                                 s3_bucket=S3_BUCKET,
                                 aws_access_key_id=AWS_ACCESS_KEY_ID,
                                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
                                 aws_region=AWS_REGION)

        try:
            # Create local directory if it doesn't exist
            os.makedirs(LOCAL_DIR, exist_ok=True)

            # Download model
            model_dir = transfer.download_model(LOCAL_DIR)

            # Upload to S3 and cleanup
            transfer.upload_to_s3(model_dir)
            shutil.rmtree(model_dir)

        except Exception as e:
            logger.error("Error in transfer process: %s", str(e))
            raise


if __name__ == "__main__":
    main()
