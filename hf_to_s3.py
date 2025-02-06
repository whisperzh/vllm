import os
import requests
import boto3
from huggingface_hub import HfApi, snapshot_download
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelTransfer:
    def __init__(self, model_id, s3_bucket, aws_access_key_id=None, aws_secret_access_key=None, aws_region=None):
        """
        Initialize the ModelTransfer class.
        
        Args:
            model_id (str): HuggingFace model ID (e.g., 'meta-llama/Llama-2-7b')
            s3_bucket (str): Name of the S3 bucket
            aws_access_key_id (str, optional): AWS access key ID. Defaults to None.
            aws_secret_access_key (str, optional): AWS secret access key. Defaults to None.
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
            region_name=aws_region
        )
        
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
        logger.info(f"Downloading model {self.model_id}...")
        
        try:
            local_dir_with_model = os.path.join(local_dir, self.model_name)
            snapshot_download(
                repo_id=self.model_id,
                local_dir=local_dir_with_model,
                local_dir_use_symlinks=False,
                token=os.getenv("HF_TOKEN")
            )
            logger.info(f"Model downloaded successfully to {local_dir_with_model}")
            return local_dir_with_model
            
        except Exception as e:
            logger.error(f"Error downloading model: {str(e)}")
            raise

    def upload_to_s3(self, local_dir):
        """
        Upload the model directory to S3.
        
        Args:
            local_dir (str): Local directory containing the model files
        """
        logger.info(f"Uploading model to S3 bucket {self.s3_bucket}...")
        
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
                    with tqdm(total=file_size, unit='B', unit_scale=True, desc=f"Uploading {filename}") as pbar:
                        self.s3_client.upload_file(
                            local_path,
                            self.s3_bucket,
                            s3_path,
                            Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
                        )
                    
                    logger.info(f"Uploaded {filename} to s3://{self.s3_bucket}/{s3_path}")
            
            logger.info("Model upload completed successfully!")
            
        except Exception as e:
            logger.error(f"Error uploading to S3: {str(e)}")
            raise

def main():
    # Configuration
    MODEL_ID = "facebook/bart-base"  # Replace with your model ID
    S3_BUCKET = "vllm-ci-model-weights"  # Replace with your S3 bucket name
    LOCAL_DIR = "~/models"  # Local directory to temporarily store the model
    
    # AWS credentials (alternatively, use AWS CLI configuration or environment variables)
    AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
    AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
    AWS_REGION = "us-west-2"
    
    # Create transfer object
    transfer = ModelTransfer(
        model_id=MODEL_ID,
        s3_bucket=S3_BUCKET,
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        aws_region=AWS_REGION
    )
    
    try:
        # Create local directory if it doesn't exist
        os.makedirs(LOCAL_DIR, exist_ok=True)
        
        # Download model
        model_dir = transfer.download_model(LOCAL_DIR)
        
        # Upload to S3
        transfer.upload_to_s3(model_dir)
        
    except Exception as e:
        logger.error(f"Error in transfer process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
