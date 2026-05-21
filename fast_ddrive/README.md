# Fast-dDrive Project Page

Static site for the Fast-dDrive paper. Layout adapted from the
[Fast-dVLM project page](https://nvlabs.github.io/Fast-dLLM/fast_dvlm/).

## Structure
- `index.html` — main page
- `asset/` — figures (PNG), demo videos (MP4)

## Local Preview
```bash
cd webpage
python3 -m http.server 8000
# open http://localhost:8000
```

## Assets sourced from the paper repo
- `fast-dDrive-teaser.png` — speed/accuracy frontier
- `fast-ddrive-pipeline.png` — overall pipeline
- `scaffold_spec.png` — Scaffold Speculative Decoding diagram
- `inference_scaling_curve.png`, `variance_reduction_curve.png` — Test-time scaling plots
- `ss_multi_rollout_wrapfig_updated.png` — Multi-rollout qualitative
- `demo_compare.mp4` — Fast-dDrive vs Qwen2.5-VL-3B AR baseline (40s)
- `demo_5hz.mp4` — 5Hz dense playback on a Waymo scene
