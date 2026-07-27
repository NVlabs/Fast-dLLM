"""Fast-dDrive single-shot inference demo.

Loads the Fast-dDrive HF release (via ``trust_remote_code=True``) and decodes
one image+prompt using one of three paths:

* ``section_diffusion``  — iterative MDM denoising over the scaffold (SD).
* ``scaffold_spec``      — scaffold-aware self-speculative decoding (SS, paper canonical).
* ``inference_scaling``  — SS with multi-trajectory rollouts (test-time scaling).
"""

import argparse

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer


_MODES = {
    "section_diffusion":  ("mdm_sample_deep_scaffold",        0.9),
    "scaffold_spec":      ("scaffold_speculative_sample",     0.0),
    "inference_scaling":  ("scaffold_spec_with_ss_multi_traj", 0.0),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="Efficient-Large-Model/Fast-dDrive",
                   help="Fast-dDrive checkpoint directory or HuggingFace id "
                        "(default: Efficient-Large-Model/Fast-dDrive on the Hugging Face Hub).")
    p.add_argument("--image", required=True, help="Path to a single image.")
    p.add_argument("--prompt", required=True, help="Text prompt.")
    p.add_argument("--mode", default="scaffold_spec", choices=sorted(_MODES),
                   help="Decoding path; scaffold_spec is the paper canonical (SS).")
    p.add_argument("--confidence_threshold", type=float, default=None,
                   help="Override the per-mode default (0.0 for scaffold_spec / "
                        "inference_scaling, 0.9 for section_diffusion).")
    args = p.parse_args()

    method_name, default_threshold = _MODES[args.mode]
    threshold = args.confidence_threshold if args.confidence_threshold is not None else default_threshold

    print(f"Loading model from {args.model_path} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16,
        device_map="cuda:0", trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(args.model_path, use_fast=False)
    processor.tokenizer = tokenizer

    image = Image.open(args.image).convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": args.prompt},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda:0")

    mask_token = "|<MASK>|"
    mask_ids = tokenizer.encode(mask_token, add_special_tokens=False)
    if len(mask_ids) != 1:
        raise ValueError(f"Expected {mask_token!r} to map to a single token, got {mask_ids}")
    mask_id = int(mask_ids[0])
    kwargs = dict(
        input_ids=inputs.input_ids,
        tokenizer=tokenizer,
        block_size=32,
        max_tokens=512,
        mask_id=mask_id,
        threshold=threshold,
    )
    if getattr(inputs, "pixel_values", None) is not None:
        kwargs["pixel_values"] = inputs.pixel_values
    if getattr(inputs, "image_grid_thw", None) is not None:
        kwargs["image_grid_thw"] = inputs.image_grid_thw

    print(f"Decoding with {method_name} (threshold={threshold}) ...", flush=True)
    with torch.inference_mode():
        out = getattr(model, method_name)(**kwargs)
    trimmed = out[0, inputs.input_ids.shape[1]:]
    print("\n--- Response ---\n" + tokenizer.decode(trimmed, skip_special_tokens=True))


if __name__ == "__main__":
    main()
