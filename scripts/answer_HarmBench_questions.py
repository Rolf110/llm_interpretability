import os
import re
from datetime import datetime
from pathlib import Path

import dotenv
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

dotenv.load_dotenv()

MODEL_NAME = "google/gemma-2b-it"
HF_DATASET_NAME = "walledai/HarmBench"
CACHE_DIR = "./cached_models/"
OUTPUT_DIR = "./output"
ANSWER_SPLITTER = "\n\n\n"
DEVICE = "cuda:0"


def sanitize_filename(name: str) -> str:
    """Sanitize names for safe filesystem use."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def get_output_filename() -> str:
    """Generate standardized output filename."""
    model_safe = sanitize_filename(MODEL_NAME)
    dataset_safe = sanitize_filename(HF_DATASET_NAME)
    timestamp = datetime.now().isoformat(timespec="seconds")
    return f"{model_safe}|{dataset_safe}|{timestamp}.txt"


def get_model_answer(model, tokenizer, question: str) -> str:
    """Generate model response with error handling."""
    try:
        chat = [{"role": "user", "content": question}]
        prompt = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
        outputs = model.generate(
            inputs.to(model.device),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
        )
        return tokenizer.decode(outputs.squeeze())
    except Exception as e:
        return f"Error occurred: {str(e)}\nDuring answering question: {question}"


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / get_output_filename()

    dataset = load_dataset(
        HF_DATASET_NAME,
        "standard",
        cache_dir=os.getenv("HarmBench_dir", None)
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        token=os.getenv("HF_TOKEN", None)
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.float16,
        revision="float16",
        cache_dir=CACHE_DIR,
        token=os.getenv("HF_TOKEN", None),
    )
    
    answers = []
    for prompt in tqdm(dataset["train"]["prompt"]): # type: ignore
        answers.append(get_model_answer(model, tokenizer, prompt))
    
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(ANSWER_SPLITTER.join(answers))
    
            
if __name__ == "__main__":
    main()