import sagemaker
from data import output_path, load_sagemaker_config, job_name,train_dataset_s3_path,val_dataset_s3_path,train_config_s3_path

# Load resources
sagemaker_session = sagemaker.Session()
bucket_name = sagemaker_session.default_bucket()
default_prefix = sagemaker_session.default_bucket_prefix
configs = load_sagemaker_config()

# Specify machine and image details
instance_type = "ml.p4d.24xlarge"
instance_count = 1

image_uri = sagemaker.image_uris.retrieve(
    framework="huggingface",
    region=sagemaker_session.boto_session.region_name,
    version="4.49.0",
    base_framework_version="pytorch2.5.1",
    instance_type=instance_type,
    image_scope="training",
)

# Import configuration variables
from sagemaker.modules.configs import (CheckpointConfig, Compute, OutputDataConfig, SourceCode, InputData, StoppingCondition)
from sagemaker.modules.distributed import Torchrun
from sagemaker.modules.train import ModelTrainer

# Define the script to be run
source_code = SourceCode(
    source_dir="./scripts",
    requirements="requirements.txt",
    entry_script="train.py",
)

# Define the compute
compute_configs = Compute(
    instance_type=instance_type,
    instance_count=instance_count,
    keep_alive_period_in_seconds=0,
)

# Define the ModelTrainer
model_trainer = ModelTrainer(
    training_image=image_uri,
    source_code=source_code,
    base_job_name=job_name,
    compute=compute_configs,
    distributed=Torchrun(),
    stopping_condition=StoppingCondition(max_runtime_in_seconds=18000),
    hyperparameters={
        "config": "/opt/ml/input/data/config/args.yaml"  # path to TRL config which was uploaded to s3
    },
    output_data_config=OutputDataConfig(s3_output_path=output_path),
    checkpoint_config=CheckpointConfig(
        s3_uri=output_path + "/checkpoint", local_path="/opt/ml/checkpoints"
    ),
)

# Define InputDataConfig paths
train_input = InputData(
    channel_name="train",
    data_source=train_dataset_s3_path, # S3 path where training data is stored
)

val_input = InputData(
    channel_name="val",
    data_source=val_dataset_s3_path, # S3 path where training data is stored
)

config_input = InputData(
    channel_name="config",
    data_source=train_config_s3_path, # S3 path where training data is stored
)
data = [train_input, val_input, config_input]

# Starting the train job with our uploaded datasets as input
model_trainer.train(input_data_config=data, wait=False)