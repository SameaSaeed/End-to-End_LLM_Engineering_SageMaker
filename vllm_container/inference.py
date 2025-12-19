import os
import json
import boto3
import pandas as pd
import requests

# Load test data
test_path = "/opt/ml/input/data/test/dataset.json"
with open(test_path, "r") as f:
    test_dataset = json.load(f)

# Endpoint info
endpoint_name = os.environ.get("SM_ENDPOINT_NAME")

sagemaker_runtime = boto3.client("sagemaker-runtime")
eval_results = []

for idx, item in enumerate(test_dataset, 1):
    print(f"Processing item {idx}")
    payload = {
        "messages": [
            {"role": "system", "content": "You are a deep-thinking AI assistant."},
            {"role": "user", "content": item["question"]}
        ],
        "max_tokens": 4096,
        "temperature": 0.4,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "do_sample": True
    }

    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    eval_results.append([item["question"], result["choices"][0]["message"]["content"]])
    print("**********************************************")

# Save results
output_path = "/opt/ml/output/eval_results.jsonl"
df = pd.DataFrame(eval_results, columns=["question", "answer"])
df.to_json(output_path, orient="records", lines=True)
print(f"Evaluation results saved to {output_path}")
