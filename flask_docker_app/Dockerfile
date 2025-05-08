# 1. CUDA-ready PyTorch base image
# FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM nvidia/cuda:12.1.1-base-ubuntu22.04
# FROM vllm/vllm-openai:latest


# 2. Set working directory
WORKDIR /app

# 3. Install required packages (Flask + Git)
RUN apt-get update && apt-get install -y \
git \
python3 \
python3-pip \
python3-dev \
&& rm -rf /var/lib/apt/lists/*
# RUN apt-get install python-pip 
RUN pip3 install -v flask

# 4. Clone the required repo (at build time)
#    Or comment this out to do it dynamically at runtime instead
RUN git clone https://github.com/whisperzh/vllm.git 

RUN cd vllm && \
VLLM_USE_PRECOMPILED=1 pip install --editable .
    # VLLM_USE_PRECOMPILED=1 
# 5. Copy your expert Flask app
COPY expert.py /app/vllm


# RUN python /app/vllm/expert.py
CMD ["python3", "./vllm/expert.py"]
