# Install transformers from source - only needed for versions <= v4.34
# pip install git+https://github.com/huggingface/transformers.git
# pip install accelerate

import torch
from transformers import pipeline


def model_query(query: str):
    pipe = pipeline(
        "text-generation",
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
    messages = [
        {"role": "user", "content": f"{query}"},
    ]
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(
        prompt,
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
    )


    output = outputs[0]["generated_text"]
    return output


if __name__ == "__main__":
    print(model_query("tell me a joke about politicians"))
