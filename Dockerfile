FROM python:3.8-slim-buster

# Install dependencies:
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio===0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Run the application:
COPY . .
CMD ["python", "main.py"]