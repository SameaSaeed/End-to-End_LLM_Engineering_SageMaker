FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Install vLLM and dependencies
RUN pip install --no-cache-dir vllm[transformers] sagemaker-inference transformers accelerate

# Copy your SageMaker entrypoint
COPY serve_vllm.py /opt/ml/code/serve_vllm.py
WORKDIR /opt/ml/code

# SageMaker environment variables
ENV SAGEMAKER_PROGRAM serve_vllm.py
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
