# Fast-dLLM
[![Project](https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages)](https://nvlabs.github.io/Fast-dLLM)
[![arXiv v1](https://img.shields.io/badge/Paper-v1-red.svg)](https://arxiv.org/abs/2505.22618)
[![arXiv v2](https://img.shields.io/badge/Paper-v2-red.svg)](https://arxiv.org/abs/2509.26328)
[![arXiv dVLM](https://img.shields.io/badge/Paper-dVLM-red.svg)](https://arxiv.org/abs/2604.06832)
<a href="https://fast-dllm.hanlab.ai"><img src="https://img.shields.io/static/v1?label=Demo&message=Fast-dLLM&color=yellow"></a> &ensp;

<h4 align="center"> ICLR 2026 </h4>
Boo

Fast-dLLM is a family of acceleration techniques for diffusion-based Large Language Models (dLLMs) and Vision-Language Models (dVLMs). This repository contains:

| | Fast-dLLM v1 | Fast-dLLM v2 | Fast-dVLM |
|---|---|---|---|
| **Paper** | [Training-free Acceleration of Diffusion LLM](https://arxiv.org/abs/2505.22618) | [Efficient Block-Diffusion LLM](https://arxiv.org/abs/2509.26328) | [Block-Diffusion VLM via Direct Conversion](https://arxiv.org/abs/2604.06832) |
| **Modality** | Text | Text | Vision + Text |
| **Approach** | Training-free inference acceleration | Block diffusion with fine-tuning | Direct AR-to-diffusion VLM conversion |
| **Backbone** | [Dream](https://github.com/dream-project/dream), [LLaDA](https://github.com/llada-project/llada) | [Qwen2.5](https://github.com/QwenLM/Qwen2.5) | [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) |
| **Key Techniques** | KV Cache + Parallel Decoding | Block Diffusion + Hierarchical Caching | Block-Size Annealing + Speculative Decoding |
| **Code** | [`v1/`](v1/) | [`v2/`](v2/) | [`fast_dvlm/`](fast_dvlm/) |
| **Model** | — | [Fast_dLLM_v2_7B](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B) | [Fast_dVLM_3B](https://huggingface.co/Efficient-Large-Model/Fast_dVLM_3B) |

## News
* (🔥 New) [2026/04/10] **Fast-dVLM** is released! Up to **6.18x speedup** over AR baseline while matching quality across 11 benchmarks. Check out our [webpage](https://nvlabs.github.io/Fast-dLLM/fast_dvlm/), [model](https://huggingface.co/Efficient-Large-Model/Fast_dVLM_3B), and [paper](https://arxiv.org/abs/2604.06832)!
* (🔥 New) [2026/01/26] **Fast-dLLM v1/v2 is accepted by ICLR-2026.** 🎉🎉🎉
* \[2025.10.08\] We have open sourced Fast-dLLM v2. Have a look at our [webpage](https://nvlabs.github.io/Fast-dLLM/v2/), [model](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_7B), and [paper](https://arxiv.org/pdf/2509.26328)!
* \[2025.08.01\] Our new online demo of Fast-dLLM: https://fast-dllm.hanlab.ai/, welcome to try!
* \[2025.07.06\] Added factor-based parallel strategy and LLaDA-1.5 evaluation in `v1/llada/eval_gsm8k.sh`.
* \[2025.07.04\] We updated our paper with latest improvements and evaluation results.
* \[2025.06.30\] Fast-dLLM has been integrated into [LLaDA-V](https://github.com/ML-GSAI/LLaDA-V). With Fast-dLLM, it accelerates the inference latency from 60s to 6s! Have a try [here](https://github.com/ML-GSAI/LLaDA-V/blob/main/train/generate_demo.py)!!

## TODOs
- \[✅\] Inference and evaluation code
- \[✅\] Training code of Fast-dLLM v2
- \[✅\] Fast-dVLM: Block-diffusion VLM
- \[🚀\] vLLM support

## Project Structure

```
Fast-dLLM/
├── v1/                     # Fast-dLLM v1: Training-free acceleration (LLM)
│   ├── dream/              #   Dream model support
│   ├── llada/              #   LLaDA model support
│   ├── requirements.txt
│   └── README.md
├── v2/                     # Fast-dLLM v2: Block diffusion (LLM)
│   ├── src/                #   LMFlow training framework
│   ├── train_scripts/      #   Fine-tuning scripts
│   ├── configs/            #   DeepSpeed configs
│   ├── generation_functions.py
│   ├── eval.py / eval_script.sh
│   ├── app.py / run_chatbot.py
│   ├── requirements.txt
│   └── README.md
├── fast_dvlm/              # Fast-dVLM: Block-diffusion VLM (chatbot, optional finetune sample, VLMEval; see fast_dvlm/README.md)
├── CONTRIBUTING.md
├── LICENSE
└── README.md               # This file
```

## Quick Start

### Fast-dLLM v1 (Training-free Acceleration)

```bash
cd v1
pip install -r requirements.txt

# LLaDA interactive chat
python llada/chat.py --gen_length 128 --steps 128 --block_size 32

# Dream evaluation
accelerate launch dream/eval.py --model dream \
    --model_args pretrained=Dream-org/Dream-v0-Base-7B,max_new_tokens=256,diffusion_steps=8,add_bos_token=true,alg=confidence_threshold,threshold=0.9,use_cache=true \
    --tasks gsm8k --num_fewshot 5 --batch_size 1
```

For full details, see [v1/README.md](v1/README.md).

### Fast-dLLM v2 (Block Diffusion)

```bash
cd v2
pip install -e .

# Gradio web demo
python app.py

# Evaluation
bash eval_script.sh
```

For full details, see [v2/README.md](v2/README.md).

### Fast-dVLM (Block-Diffusion VLM)

```bash
cd fast_dvlm
pip install -r requirements.txt

# Quick inference
python run_chatbot.py \
    --model-name Efficient-Large-Model/Fast_dVLM_3B \
    --image path/to/image.jpg \
    --prompt "Describe this image in detail."

# Interactive mode
python run_chatbot.py
```

**Fine-tuning (optional example):** multimodal MDM training uses DeepSpeed + the LMFlow fork under [`third_party/`](third_party/) (the launcher sets `PYTHONPATH` for you). Download [ALLaVA-4V](https://huggingface.co/datasets/FreedomIntelligence/ALLaVA-4V) with `fast_dvlm/data/download_example_dataset.sh`, then run `bash fast_dvlm/train_scripts/finetune_multimodal_example.sh` from the repo root—see [Fine-tuning (example launcher)](fast_dvlm/README.md#fine-tuning-example-launcher) in [fast_dvlm/README.md](fast_dvlm/README.md).

For full details, see [fast_dvlm/README.md](fast_dvlm/README.md).

## Contributing

Issues and Pull Requests are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the Apache License 2.0. See the [LICENSE](LICENSE) file for details.

## Citation

If you find this work useful, please cite our papers:

```bibtex
@misc{wu2026fastdvlmefficientblockdiffusionvlm,
      title={Fast-dVLM: Efficient Block-Diffusion VLM via Direct Conversion from Autoregressive VLM},
      author={Chengyue Wu and Shiyi Lan and Yonggan Fu and Sensen Gao and Jin Wang and Jincheng Yu and Jose M. Alvarez and Pavlo Molchanov and Ping Luo and Song Han and Ligeng Zhu and Enze Xie},
      year={2026},
      eprint={2604.06832},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.06832},
}
@misc{wu2025fastdllmv2efficientblockdiffusion,
      title={Fast-dLLM v2: Efficient Block-Diffusion LLM}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Shizhe Diao and Yonggan Fu and Zhijian Liu and Pavlo Molchanov and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2509.26328},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.26328}, 
}
@misc{wu2025fastdllmtrainingfreeaccelerationdiffusion,
      title={Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding}, 
      author={Chengyue Wu and Hao Zhang and Shuchen Xue and Zhijian Liu and Shizhe Diao and Ligeng Zhu and Ping Luo and Song Han and Enze Xie},
      year={2025},
      eprint={2505.22618},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.22618}, 
}
```

## Acknowledgements

We would like to thank the authors of [LLaDA](https://github.com/llada-project/llada) and [Dream](https://github.com/dream-project/dream) for their excellent work and open-source contributions. We thank [Qwen2.5](https://github.com/QwenLM/Qwen2.5) and [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL) for the base model architectures and [LMFlow](https://github.com/OptimalScale/LMFlow) for the training framework.
