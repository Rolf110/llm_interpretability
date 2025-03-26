import os
import re
import functools
from datetime import datetime
from pathlib import Path

import dotenv
from datasets import load_dataset
from tqdm.auto import tqdm
from transformer_lens import HookedTransformer

dotenv.load_dotenv()

MODEL_NAME = "google/gemma-2b-it"
HF_DATASET_NAME = "walledai/HarmBench"
CACHE_DIR = "./cached_models/"
OUTPUT_DIR = "./output"
ANSWER_SPLITTER = "\n\n\n"
DEVICE = "cuda:0"

safety_heads = [
    (0, 4),
    (0, 6),
    (1, 3),
    (3, 1),
    (3, 2),
    (3, 5),
    (4, 5),
    (4, 6),
    (4, 7),
    (5, 2),
    (5, 6),
    (5, 7),
    (6, 1),
    (6, 4),
    (6, 5),
    (7, 1),
    (7, 2),
    (7, 3),
    (9, 1),
    (9, 2),
    (9, 3),
    (10, 1),
    (11, 1),
    (11, 6),
    (12, 2),
    (12, 3),
    (13, 1),
    (13, 2),
    (14, 0),
    (14, 1),
    (14, 5),
    (15, 2),
    (15, 4),
    (15, 5),
    (15, 6),
    (15, 7),
    (16, 1),
    (16, 3),
    (16, 5),
    (16, 7),
    (17, 2),
    (17, 3),
    (17, 7),
]


def prepare_prompt(prompt: str) -> str:
    """Prepare the prompt for the model."""
    harmful_prompt = "<bos><start_of_turn>user\n"
    harmful_prompt += prompt
    harmful_prompt += "\n<start_of_turn>model"
    return harmful_prompt


def sanitize_filename(name: str) -> str:
    """Sanitize names for safe filesystem use."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", name)


def get_output_filename() -> str:
    """Generate standardized output filename."""
    model_safe = sanitize_filename(MODEL_NAME)
    dataset_safe = sanitize_filename(HF_DATASET_NAME)
    timestamp = datetime.now().isoformat(timespec="seconds")
    return f"zero_ablation|{model_safe}|{dataset_safe}|{timestamp}.txt"


def head_ablation_hook(
    attn_result, 
    hook, 
    head_index_to_ablate
):
    """
    Hook function that zeros out a specific attention head.
    
    Args:
        attn_result: Attention result tensor of shape [batch, seq, n_heads, d_model]
        hook: HookPoint
        head_index_to_ablate: Index of the head to ablate
    
    Returns:
        Modified attention result with the specified head zeroed out
    """
    attn_result[:, :, head_index_to_ablate, :] = 0.0
    return attn_result


def run_model_with_ablated_heads(
    model, 
    prompt, 
    heads_to_ablate,
    **generation_kwargs
):
    """
    Runs the model with specific attention heads ablated.
    
    Args:
        model: HookedTransformer model
        prompt: Input text or tokens
        heads_to_ablate: List of (layer, head) tuples to ablate
        **generation_kwargs: Additional keyword arguments for model.generate
    
    Returns:
        Model output with the specified heads ablated
    """
    # Create hooks for each head to ablate
    model.reset_hooks()
    
    # Add hooks for each head to ablate
    hooks = []
    for layer, head in heads_to_ablate:
        # Use functools.partial to create a hook function with the head fixed
        hook_fn = functools.partial(head_ablation_hook, head_index_to_ablate=head)
        # Add hook to the appropriate hook point
        hook_point = f"blocks.{layer}.attn.hook_z"
        hook = model.add_hook(hook_point, hook_fn)
        hooks.append(hook)
    
    # For generation, we need to use model.generate
    if isinstance(prompt, str):
        tokens = model.to_tokens(prompt)
    else:
        tokens = prompt
    
    # Generate text with the hooks applied
    output = model.generate(
        tokens,
        verbose=False,
        **generation_kwargs
    ).squeeze()
    
    model.reset_hooks()
    
    return model.to_string(output)


def main():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    output_path = Path(OUTPUT_DIR) / get_output_filename()

    dataset = load_dataset(
        HF_DATASET_NAME,
        "standard",
        cache_dir=os.getenv("HarmBench_dir", None)
    )
    
    model = HookedTransformer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    
    answers = []
    for prompt in tqdm(dataset["train"]["prompt"]): # type: ignore
        harmful_prompt = prepare_prompt(prompt)
        
        # Generate text with the ablated safety heads
        output = run_model_with_ablated_heads(
            model, 
            harmful_prompt, 
            heads_to_ablate=safety_heads,
            max_new_tokens=256
        )
        answers.append(output)
    
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(ANSWER_SPLITTER.join(answers))
    
            
if __name__ == "__main__":
    main()