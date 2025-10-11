from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os, csv, torch, argparse
from transformers import AutoTokenizer
from utils.similarity import LayerTap, cosine_matrix, cosine_diag

# import Dream model (names may differ; adjust if needed)
from dream.model.modeling_dream import DreamModel

def get_decoder_layers_dream(model):
    layers = []
    for name, mod in model.named_modules():
        if type(mod).__name__ in {"DreamDecoderLayer"}:
            layers.append(mod)
    if not layers:
        raise RuntimeError("Could not find Dream decoder layers")
    return layers

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float16

    # Replace with the correct Dream checkpoint id / loader in this repo
    model_id = "HKUNLP/Dream-<ID>"   # <-- if the repo loads locally, adapt accordingly
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = DreamModel.from_pretrained(model_id, trust_remote_code=True, torch_dtype=dtype).to(device)
    model.eval()

    layers = get_decoder_layers_dream(model)
    tap = LayerTap(layers, pool="last")

    prev = None
    os.makedirs(args.out_dir, exist_ok=True)
    across_path = os.path.join(args.out_dir, "across_steps_cosine.csv")

    with open(across_path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step"] + [f"layer_{i}" for i in range(len(layers))])

        def on_step(step):
            nonlocal prev
            A = tap.stacked()
            torch.save(cosine_matrix(A), os.path.join(args.out_dir, f"within_step_cosine_step{step}.pt"))
            if prev is not None:
                w.writerow([step] + cosine_diag(prev, A).tolist())
            prev = A
            tap.clear()

        inputs = tok(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            # Whatever Dream's generation entrypoint is that accepts num_diffusion_steps
            _ = model.generate(
                inputs["input_ids"],
                max_new_tokens=1,
                use_cache=True,
                num_diffusion_steps=args.steps,
                block_size=args.block_size,
                temperature=0.0,
                step_callback=on_step,      # ✦ from Step 2A in Dream generation
            )

    tap.remove()
    print(f"[Dream] wrote similarity logs in {args.out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default="logs/dream_baseline")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--prompt", type=str, default="Explain diffusion decoding briefly.")
    args = ap.parse_args()
    main(args)
