FROM python:3.8-slim-buster

# Install dependencies:
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio===0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git

# Run the application:
COPY . .
ENTRYPOINT ["python", "mrf_trainer.py"]