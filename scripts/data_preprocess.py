import boto3
import sagemaker
from sagemaker.s3 import S3Uploader
from sagemaker.config import load_sagemaker_config
import os
import shutil
import pandas as pd
from random import randint
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

# Set up 3rd party environment variables
os.environ["HF_TOKEN"] = "<HF_TOKEN>"
os.environ["model_id"] = "meta-llama/Llama-3.1-8B-Instruct"
os.environ["mlflow_uri"] = ""
os.environ["mlflow_experiment_name"] = "llama-31-8b-sft"

# Set up AWS native resources
sagemaker_session = sagemaker.Session()
s3_client = boto3.client('s3')
bucket_name = sagemaker_session.default_bucket()
job_name = f"train-{os.environ["model_id"].split('/')[-1].replace('.', '-')}-sft"
default_prefix = sagemaker_session.default_bucket_prefix
if default_prefix:
    input_path = f"{default_prefix}/datasets/llm-fine-tuning-modeltrainer-sft"
    output_path = f"s3://{bucket_name}/{default_prefix}/{job_name}"
else:
    input_path = f"datasets/llm-fine-tuning-modeltrainer-sft"
    output_path = f"s3://{bucket_name}/{job_name}"

train_dataset_s3_path = f"s3://{bucket_name}/{input_path}/train/dataset.json"
val_dataset_s3_path = f"s3://{bucket_name}/{input_path}/val/dataset.json"
model_yaml = "args.yaml"
train_config_s3_path = S3Uploader.upload(local_path=model_yaml, desired_s3_uri=f"{input_path}/config")
os.remove("./args.yaml")

# Load and split dataset
from datasets import load_dataset
dataset = load_dataset("UCSC-VLAA/MedReason", split="train[:10000]")
df = pd.DataFrame(dataset)
train, val = train_test_split(df, test_size=0.1, random_state=42)
train, test = train_test_split(train, test_size=10, random_state=42)
train_dataset = Dataset.from_pandas(train)
val_dataset = Dataset.from_pandas(val)
test_dataset = Dataset.from_pandas(test)
dataset = DatasetDict({"train": train_dataset, "val": val_dataset})

# Preprocess dataset
def prepare_dataset(sample):
    tokenizer = AutoTokenizer.from_pretrained(os.environ["model_id"])
    tokenizer.eos_token = "<|eot_id|>"
    tokenizer.eos_token_id = 128009
    tokenizer.pad_token = tokenizer.eos_token
    system_text = (
        "You are a deep-thinking AI assistant.\n\n"
        "For every user question, first write your thoughts and reasoning inside <think>...</think> tags, then provide your answer."
    )

    messages = []

    messages.append({"role": "system", "content": system_text})
    messages.append({"role": "user", "content": sample["question"]})
    messages.append(
        {
            "role": "assistant",
            "content": f"<think>\n{sample['reasoning']}\n</think>\n{sample['answer']}",
        }
    )

    sample["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return sample

train_dataset = dataset["train"].map(prepare_dataset, remove_columns=list(train_dataset.features))
val_dataset = dataset["val"].map(prepare_dataset, remove_columns=list(val_dataset.features))

# Save pre-processed datasets to s3
train_dataset.to_json("./data/train/dataset.json", orient="records")
val_dataset.to_json("./data/val/dataset.json", orient="records")
s3_client.upload_file("./data/train/dataset.json", bucket_name, f"{input_path}/train/dataset.json")
s3_client.upload_file("./data/val/dataset.json", bucket_name, f"{input_path}/val/dataset.json")
shutil.rmtree("./data")