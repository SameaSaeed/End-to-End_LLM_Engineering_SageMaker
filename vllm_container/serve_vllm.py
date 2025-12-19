import os
from vllm import LLM, SamplingParams
from fastapi import FastAPI, Request
from pydantic import BaseModel
import uvicorn

model_path = os.environ.get("HF_MODEL_ID", "/opt/ml/model")
llm = LLM(model=model_path)

app = FastAPI()

class InferenceRequest(BaseModel):
    messages: list
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

@app.post("/invocations")
async def predict(request: InferenceRequest):
    prompt = ""
    for msg in request.messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            prompt += f"[SYSTEM]: {content}\n"
        elif role == "user":
            prompt += f"[USER]: {content}\n"
        elif role == "assistant":
            prompt += f"[ASSISTANT]: {content}\n"

    sampling_params = SamplingParams(
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        max_tokens=request.max_tokens,
    )

    outputs = llm.generate(prompt, sampling_params=sampling_params)
    response_text = outputs[0].text

    return {"choices": [{"message": {"role": "assistant", "content": response_text}}]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
