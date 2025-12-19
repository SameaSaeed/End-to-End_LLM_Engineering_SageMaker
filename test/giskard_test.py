import os
import pandas as pd
import boto3
import sagemaker
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from giskard import Model, Dataset, TestSuite

# ----------------------------
# Configuration
# ----------------------------
sagemaker_session = sagemaker.Session()
bucket_name = sagemaker_session.default_bucket()
endpoint_name = os.environ.get("LLAMA_ENDPOINT", "your-llama-endpoint")

# S3 path to your test dataset (JSON/JSONL)
test_dataset_s3_path = f"s3://{bucket_name}/datasets/llm-fine-tuning-modeltrainer-sft/test/dataset.json"

# ----------------------------
# Download test dataset from S3
# ----------------------------
s3_client = boto3.client("s3")
local_test_path = "/tmp/test_dataset.json"
bucket, key = test_dataset_s3_path.replace("s3://", "").split("/", 1)
s3_client.download_file(bucket, key, local_test_path)

test_df = pd.read_json(local_test_path)

# ----------------------------
# SageMaker Endpoint Predictor
# ----------------------------
predictor = sagemaker.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)

def llm_predict(samples):
    results = []
    for sample in samples:
        messages = [
            {"role": "system", "content": "You are a deep-thinking AI assistant."},
            {"role": "user", "content": sample["question"]},
        ]
        response = predictor.predict({
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.4,
            "top_p": 0.9,
        })
        results.append(response["choices"][0]["message"]["content"])
    return results

llm_model = Model.from_custom_model(
    name="Finetuned Llama",
    model_type="text_generation",
    model=llm_predict,
    inputs=[{"name": "question", "type": "text"}],
    outputs=[{"name": "answer", "type": "text"}],
)

# ----------------------------
# Giskard Dataset
# ----------------------------
giskard_dataset = Dataset.from_pandas(
    test_df,
    name="llama-test-set",
    target="answer"
)

# ----------------------------
# Define Tests
# ----------------------------
tests = TestSuite(name="Llama Reasoning Tests")

# Test 1: All answers are non-empty
tests.add_assertion(
    lambda df: df["answer"].str.strip().astype(bool).all(),
    "All answers are non-empty"
)

# Test 2: All answers contain reasoning tag <think>
tests.add_assertion(
    lambda df: df["answer"].str.contains("<think>").all(),
    "All answers contain <think> tags"
)

# Test 3: Answers are under 2048 tokens (optional)
tests.add_assertion(
    lambda df: df["answer"].str.split().str.len().lt(2048).all(),
    "All answers are under 2048 tokens"
)

# ----------------------------
# Run Tests
# ----------------------------
results = tests.run(llm_model, giskard_dataset)
print(results)

# ----------------------------
# Save artifacts
# ----------------------------
llm_model.save("finetuned_llama_model.giskard")
giskard_dataset.save("llama_test_dataset.giskard")
