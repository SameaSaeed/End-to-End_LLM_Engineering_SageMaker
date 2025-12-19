import sagemaker
from deploy_finetuned_llama import model_id, endpoint_name
from data import test_dataset

sagemaker_session = sagemaker.Session()

predictor = sagemaker.Predictor(
    endpoint_name=endpoint_name,
    sagemaker_session=sagemaker_session,
    serializer=sagemaker.serializers.JSONSerializer(),
    deserializer=sagemaker.deserializers.JSONDeserializer(),
)

import pandas as pd

eval_dataset = []

index = 1
for el in test_dataset:
    print("Processing item ", index)
    system_text = (
        "You are a deep-thinking AI assistant.\n\n"
        "For every user question, first write your thoughts and reasoning inside <think>...</think> tags, then provide your answer."
    )

    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": el["question"]},
    ]

    response = predictor.predict(
        {
            "messages": messages,
            "max_tokens": 4096,
            "stop": ["<|eot_id|>", "<|end_of_text|>"],
            "temperature": 0.4,
            "top_p": 0.9,
            "repetition_penalty": 1.15,
            "no_repeat_ngram_size": 3,
            "do_sample": True,
        }
    )

    eval_dataset.append([el["question"], response["choices"][0]["message"]["content"]])

    index += 1

    print("**********************************************")

eval_dataset_df = pd.DataFrame(
    eval_dataset, columns=["question", "answer"]
)

eval_dataset_df.to_json(
    "./eval_dataset_results.jsonl", orient="records", lines=True
)