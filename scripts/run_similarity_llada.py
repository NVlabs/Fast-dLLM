from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, csv, torch, argparse
from transformers import AutoTokenizer
from utils.similarity import LayerTap, cosine_matrix, cosine_diag

# use the repo's generation path so step_callback is honored
from llada.generate import generate_with_dual_cache  # or generate / generate_with_prefix_cache

# import LLaDA model
from llada.model.modeling_llada import LLaDAModelLM


def get_decoder_layers_llada(hf_model):
    """
    Return a flat list of transformer blocks from LLaDA, whether they are stored
    in `transformer.blocks` or grouped in `transformer.block_groups`.
    """
    # unwrap the HF wrapper -> internal LLaDAModel
    core = hf_model.model if hasattr(hf_model, "model") else hf_model

    tr = getattr(core, "transformer", None)
    if tr is None:
        raise RuntimeError("LLaDA core model has no `transformer` module")

    # Case 1: plain list of blocks
    if hasattr(tr, "blocks"):
        layers = list(tr.blocks)
        if len(layers) > 0:
            return layers

    # Case 2: blocks grouped into block_groups
    if hasattr(tr, "block_groups"):
        layers = []
        for bg in tr.block_groups:  # each is an nn.ModuleList
            layers.extend(list(bg))  # append blocks inside the group
        if len(layers) > 0:
            return layers

    # Fallback: walk modules and grab any module with LLaDABlock in its class name
    layers = [
        m
        for m in core.modules()
        if m.__class__.__name__.startswith("LLaDA") and "Block" in m.__class__.__name__
    ]
    if len(layers) > 0:
        return layers

    raise RuntimeError("Could not find LLaDA decoder layers")


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 on CUDA, float32 on CPU (avoid fp16 on CPU)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    model_id = "GSAI-ML/LLaDA-8B-Instruct"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = (
        LLaDAModelLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype)
        .to(device)
        .eval()
    )

    layers = get_decoder_layers_llada(model)
    tap = LayerTap(layers, pool="last")

    # callback that fires once per diffusion step
    prev = None
    step_idx = -1
    across_path = os.path.join(args.out_dir, "across_steps_cosine.csv")
    with open(across_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["step"] + [f"layer_{i}" for i in range(len(layers))])

        def on_step(step):
            nonlocal prev, step_idx
            step_idx = step
            A = tap.stacked()  # [L, H] pooled activations (one row per layer)

            # within-step full layer-by-layer cosine -> save per-step tensor
            cos = cosine_matrix(A)
            torch.save(
                cos, os.path.join(args.out_dir, f"within_step_cosine_step{step}.pt")
            )

            # across-steps diagonal cosine (layer i at step t vs layer i at step t-1)
            if prev is not None:
                d = cosine_diag(prev, A).tolist()
                w.writerow([step] + d)

            prev = A
            tap.clear()

        # Build prompt (plain text is OK with Instruct; chat template optional)
        prompt = args.prompt
        inputs = tok(prompt, return_tensors="pt").to(device)

        # Run the repo's LLaDA generator so `step_callback` is actually used
        with torch.no_grad():
            _tokens, _nfe = generate_with_dual_cache(
                model,
                inputs["input_ids"],
                steps=args.steps,
                gen_length=args.gen_length,
                block_length=args.block_size,
                temperature=0.0,
                remasking="low_confidence",
                step_callback=on_step,  # ✦ our hook from Step 2B
            )

    tap.remove()
    print(f"[LLaDA] wrote similarity logs in {args.out_dir}")
    print(f"- within-step: {args.out_dir}/within_step_cosine_step*.pt")
    print(f"- across-steps: {across_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="logs/llada_baseline")
    ap.add_argument("--steps", type=int, default=8, help="total diffusion steps")
    ap.add_argument(
        "--gen-length", type=int, default=64, help="number of tokens to generate"
    )
    ap.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="semi-autoregressive block length (must divide gen-length)",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default="Explain diffusion decoding briefly.",
    )
    args = ap.parse_args()
    main(args)
