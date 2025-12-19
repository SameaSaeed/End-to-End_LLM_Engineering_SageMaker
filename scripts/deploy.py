import sagemaker
from sagemaker import get_execution_role
from sagemaker.huggingface import get_huggingface_llm_image_uri
from sagemaker.huggingface import HuggingFaceModel
from load_finetuned_llama import *

job_name = get_last_job_name(job_prefix)

instance_count = 1
instance_type = "ml.g5.12xlarge"
health_check_timeout = 700

image_uri = get_huggingface_llm_image_uri(
    "huggingface",
    session=sagemaker_session,
    version="3.2",
)

if default_prefix:
    model_data = f"s3://{bucket_name}/{default_prefix}/{job_prefix}/{job_name}/output/model.tar.gz"
else:
    model_data = f"s3://{bucket_name}/{job_prefix}/{job_name}/output/model.tar.gz"

model = HuggingFaceModel(
    image_uri=image_uri,
    model_data=model_data,
    role=get_execution_role(),
    env={
        "HF_MODEL_ID": "/opt/ml/model",  # Path to the model in the container
        "SM_NUM_GPUS": "4",  # Number of GPU used per replica
        "MAX_INPUT_LENGTH": "8000",  # Max length of input text
        "MAX_TOTAL_TOKENS": "8096",  # Max length of the generation (including input text)
        "MAX_BATCH_PREFILL_TOKENS": "16182",  # Limits the number of tokens that can be processed in parallel during the generation
        "MESSAGES_API_ENABLED": "true",  # Enable the OpenAI Messages API
    },
)

endpoint_name = f"{model_id.split('/')[-1].replace('.', '-')}-tgi"

predictor = model.deploy(
    endpoint_name=endpoint_name,
    initial_instance_count=instance_count,
    instance_type=instance_type,
    container_startup_health_check_timeout=health_check_timeout,
    model_data_download_timeout=3600
)