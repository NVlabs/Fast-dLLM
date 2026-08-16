#!/usr/bin/env python3
"""Fast-dDrive — Waymo E2E batch inference.

Runs one of three decoding paths over a Waymo validation JSON and writes
``predictions.json`` + ``metrics.json``. The output ``predictions.json`` is
consumed by :file:`evaluate_waymo_metrics.py` for official ADE / RFS scoring.

Usage::

    python fast_ddrive/eval/batch_inference.py \\
        --model_path /path/to/fast_ddrive_ckpt \\
        --eval_json  /path/to/waymo_val.json \\
        --image_root /path/to/image_root \\
        --output_dir eval_outputs/run0 \\
        --mode scaffold_spec \\
        --num_gpus 8

See :file:`../run_eval.sh` for the canonical launcher.
"""

import argparse
import json
import os
import re
import sys
import time
import traceback
import types
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

# The model (`Fast_dLLM_Qwen2_5_VLForConditionalGeneration`) and its
# inference-time decoding paths (Section Diffusion, Scaffold Spec,
# Scaffold Spec multi-rollout) are loaded directly from the HF release via
# ``trust_remote_code=True``; no local registration step is needed.


# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def _resolve_single_image_path(image_path: str, image_root: str) -> str:
    rel = image_path
    for prefix in ("dataset/", "dataset\\"):
        if rel.startswith(prefix):
            rel = rel[len(prefix):]
            break
    return os.path.join(image_root, rel)


def resolve_image_path(image_field, image_root: str):
    """Resolve image path(s). Supports both str and list[str]."""
    if isinstance(image_field, list):
        return [_resolve_single_image_path(p, image_root) for p in image_field]
    return _resolve_single_image_path(image_field, image_root)


def _extract_float_pairs(s: str) -> Optional[List[List[float]]]:
    pairs = re.findall(r'\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]', s)
    if not pairs:
        return None
    return [[float(a), float(b)] for a, b in pairs]


def parse_trajectory(response: str) -> Optional[List[List[float]]]:
    if not response:
        return None

    try:
        obj = json.loads(response)
        traj = obj.get("trajectory")
        if traj is not None:
            if isinstance(traj, str):
                result = _extract_float_pairs(traj)
                if result:
                    return result
                traj = json.loads(traj)
            if isinstance(traj, list) and len(traj) > 0:
                return [[float(p[0]), float(p[1])] for p in traj]
    except Exception:
        pass

    patterns = [
        r'"trajectory"\s*:\s*"(\[\[.*?\]\])"',
        r'"trajectory"\s*:\s*(\[\[.*?\]\])',
        # Handle <|mdm_start|>/<|mdm_end|> wrapped trajectories in corrupted JSON
        r'"trajectory"\s*:\s*"[^"]*?(\[\[.*?\]\])[^"]*?"',
        r'trajectory[^[]*(\[\[[^\]]*\]\])',
    ]
    for pat in patterns:
        m = re.search(pat, response, re.DOTALL)
        if m:
            result = _extract_float_pairs(m.group(1))
            if result:
                return result
            try:
                traj = json.loads(m.group(1))
                return [[float(p[0]), float(p[1])] for p in traj]
            except Exception:
                pass

    # Last resort: extract any float pairs from the raw response
    result = _extract_float_pairs(response)
    if result and len(result) >= 3:
        return result

    return None


def parse_full_output(response: str) -> Optional[Dict]:
    if not response:
        return None
    try:
        return json.loads(response)
    except Exception:
        pass
    m = re.search(r'\{.*\}', response, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def compute_trajectory_metrics(
    pred_traj: Optional[List[List[float]]],
    gt_traj: List[List[float]],
) -> Dict[str, Any]:
    if pred_traj is None or len(pred_traj) == 0:
        return {"ade": None, "fde": None, "valid": False, "num_points": 0}

    pred = np.array(pred_traj, dtype=np.float64)
    gt   = np.array(gt_traj,   dtype=np.float64)

    n = min(len(pred), len(gt))
    if n == 0:
        return {"ade": None, "fde": None, "valid": False, "num_points": 0}

    pred, gt = pred[:n], gt[:n]
    dists = np.linalg.norm(pred - gt, axis=1)
    return {
        "ade":        float(dists.mean()),
        "fde":        float(dists[-1]),
        "valid":      True,
        "num_points": n,
    }


# ══════════════════════════════════════════════════════════════════
#  模型加载 & 推理
# ══════════════════════════════════════════════════════════════════

class WaymoEvaluatorSpeculative:
    """加载模型并使用 self-speculative decoding 推理。"""

    def __init__(
        self,
        model_path: str,
        base_model: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        block_size: Optional[int] = None,
        sub_block_size: Optional[int] = None,
        confidence_threshold: float = 0.0,
        use_mdm: int = 1,
        use_quadratic: int = 0,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = 50,
        section_block_steps: Optional[dict] = None,
        section_token_budgets: Optional[dict] = None,
    ):
        self.device = device
        self.is_mdm = False
        self.block_size = block_size
        self.sub_block_size = sub_block_size
        self.mask_id = None
        self.confidence_threshold = confidence_threshold
        self.use_mdm = use_mdm
        self.use_quadratic = use_quadratic
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.section_block_steps = section_block_steps
        self.section_token_budgets = section_token_budgets
        self._init_dav_config()

        print(f"[{device}] 加载 processor: {base_model}", flush=True)
        proc_kwargs = {}
        if min_pixels is not None:
            proc_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels

        self.processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code=True, **proc_kwargs
        )

        # Load the released HF model.  ``trust_remote_code=True`` pulls
        # ``Fast_dLLM_Qwen2_5_VLForConditionalGeneration`` (defined in the HF
        # repo's ``modeling.py``) and attaches the three decoding paths
        # (``mdm_sample_deep_scaffold`` / ``scaffold_speculative_sample`` /
        # ``scaffold_spec_with_ss_multi_traj``) onto the model class at import
        # time, so no manual method binding is needed here.
        print(f"[{device}] 加载模型: {model_path}", flush=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        ).eval()

        cfg = self.model.config
        if hasattr(cfg, "bd_size") and cfg.bd_size is not None:
            self.is_mdm = True
            if self.block_size is None:
                self.block_size = cfg.bd_size
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.processor.tokenizer = tokenizer
            mask_ids = tokenizer.encode("|<MASK>|", add_special_tokens=False)
            self.mask_id = mask_ids[0] if mask_ids else 151665
            self._tokenizer = tokenizer
            print(
                f"[{device}] 检测到 MDM 模型，bd_size={self.block_size}，"
                f"mask_id={self.mask_id}",
                flush=True,
            )
            print(
                f"[{device}] SECTION_VERSION={os.environ.get('SECTION_VERSION', '(unset)')}, "
                f"confidence_threshold={self.confidence_threshold}",
                flush=True,
            )
        else:
            print(f"[{device}] 检测到标准 Qwen2.5-VL / AR 模型", flush=True)
            if self.use_mdm == 0:
                self.is_mdm = False

    # ── spec_dav helpers ────────────────────────────────────────────

    def _init_dav_config(self):
        """Parse DaV parameters from environment variables (called once at init)."""
        self._dav_num_candidates = int(os.environ.get("DAV_NUM_CANDIDATES", "4"))
        self._dav_greedy_margin = float(os.environ.get("DAV_GREEDY_MARGIN", "0.05"))
        self._dav_scoring_mode = os.environ.get("DAV_SCORING_MODE", "denoising")
        self._dav_temperatures_list = [float(x) for x in os.environ.get("DAV_TEMPERATURES", "0.0,0.3,0.5,0.7").split(",")]
        self._dav_noise_levels_list = [float(x) for x in os.environ.get("DAV_NOISE_LEVELS", "0.3,0.6").split(",")]
        _sw_str = os.environ.get("DAV_SCORE_WEIGHTS", "")
        self._dav_score_weights = json.loads(_sw_str) if _sw_str else None
        self._last_dav_stats = None

    def _build_value_mask(self, traj_tokens):
        """Classify trajectory tokens as value (True) or structural (False).

        Structural tokens contain only characters from "[], \\n\\t" and are
        kept frozen during re-denoising and scoring.
        """
        structural_chars = set("[], \n\t")
        traj_len = len(traj_tokens)
        texts = self._tokenizer.batch_decode([[t.item()] for t in traj_tokens])
        value_mask = torch.zeros(traj_len, dtype=torch.bool, device=self.device)
        for i, tok_text in enumerate(texts):
            if not all(c in structural_chars for c in tok_text):
                value_mask[i] = True
        return value_mask

    def _find_trajectory_token_range(self, output_ids, prompt_len):
        """Find the token range of trajectory coordinate content in the output.

        Returns (abs_start, abs_end) as absolute positions in output_ids,
        or (None, None) if trajectory cannot be found.
        """
        gen_ids = output_ids[0, prompt_len:].tolist()
        full_text = self._tokenizer.decode(gen_ids, skip_special_tokens=False)

        traj_key = '"trajectory"'
        key_idx = full_text.rfind(traj_key)
        if key_idx < 0:
            return None, None

        content_start = full_text.find("[[", key_idx)
        if content_start < 0:
            content_start = full_text.find("[+", key_idx)
        if content_start < 0:
            return None, None

        # Build cumulative text-length array via per-token decode (single pass)
        token_texts = self._tokenizer.batch_decode([[t] for t in gen_ids])
        cum_len = 0
        traj_token_start = len(gen_ids)  # fallback
        for i, tt in enumerate(token_texts):
            cum_len += len(tt)
            if cum_len > content_start:
                traj_token_start = i
                break

        # Skip trailing special tokens (e.g. <|im_end|>, <|endoftext|>)
        traj_token_end = len(gen_ids)
        _stop_tokens = (151645, 151643)
        while traj_token_end > traj_token_start and gen_ids[traj_token_end - 1] in _stop_tokens:
            traj_token_end -= 1

        return prompt_len + traj_token_start, prompt_len + traj_token_end

    @torch.no_grad()
    def _redenoise_trajectory(
        self, output_ids, traj_start, traj_end, temperature,
        block_size, pixel_values=None, image_grid_thw=None,
        mask_ratio=0.3,
    ):
        """Generate a trajectory variant by lightly perturbing greedy tokens.

        Randomly masks ``mask_ratio`` of value tokens (digits, signs, decimals)
        while keeping all structural tokens frozen, then fills the masked
        positions in a single forward pass with temperature sampling.

        This preserves trajectory format while creating diversity.
        """
        import torch.nn.functional as F

        traj_len = traj_end - traj_start
        greedy_traj = output_ids[0, traj_start:traj_end].clone()

        value_mask = self._build_value_mask(greedy_traj)
        if value_mask.sum().item() == 0:
            return greedy_traj

        # Randomly select mask_ratio of value tokens to re-denoise
        noise = torch.rand(traj_len, device=self.device)
        should_mask = (noise < mask_ratio) & value_mask
        if should_mask.sum().item() == 0:
            return greedy_traj

        x_t = output_ids.clone()
        x_t[0, traj_start:traj_end][should_mask] = self.mask_id

        self.model.bd_size = block_size

        fwd_kwargs = dict(input_ids=x_t, use_cache=False)
        if pixel_values is not None:
            fwd_kwargs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            fwd_kwargs["image_grid_thw"] = image_grid_thw

        logits = self.model.forward(**fwd_kwargs).logits

        traj_logits = torch.cat([
            logits[:, traj_start - 1 : traj_start, :],
            logits[:, traj_start : traj_end - 1, :],
        ], dim=1)

        # Fill masked positions with temperature sampling
        result = greedy_traj.clone()
        if temperature > 0:
            scaled = traj_logits[0] / temperature
            probs = F.softmax(scaled, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            sampled = traj_logits[0].argmax(dim=-1)

        result[should_mask] = sampled[should_mask]
        return result

    @torch.no_grad()
    def _score_trajectory_denoising(
        self, output_ids, traj_tokens, traj_start, traj_end,
        noise_levels, pixel_values=None, image_grid_thw=None,
        block_size=32, samples_per_level=1,
    ):
        """Score a trajectory candidate via denoising likelihood.

        Only value tokens (numeric) are masked for scoring; structural tokens
        ([, ], comma, space) are kept frozen.  Higher score → better candidate.
        """
        import torch.nn.functional as F

        traj_len = traj_end - traj_start
        total_log_prob = 0.0
        total_count = 0

        value_mask = self._build_value_mask(traj_tokens)
        if value_mask.sum().item() == 0:
            return 0.0

        self.model.bd_size = block_size

        for t in noise_levels:
            for _ in range(samples_per_level):
                noise = torch.rand(traj_len, device=self.device)
                should_mask = (noise < t) & value_mask
                n_masked = should_mask.sum().item()
                if n_masked == 0:
                    continue

                x_score = output_ids.clone()
                x_score[0, traj_start:traj_end] = traj_tokens.clone()
                x_score[0, traj_start:traj_end][should_mask] = self.mask_id

                fwd_kwargs = dict(input_ids=x_score, use_cache=False)
                if pixel_values is not None:
                    fwd_kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    fwd_kwargs["image_grid_thw"] = image_grid_thw

                logits = self.model.forward(**fwd_kwargs).logits

                traj_logits = torch.cat([
                    logits[:, traj_start - 1 : traj_start, :],
                    logits[:, traj_start : traj_end - 1, :],
                ], dim=1)

                log_probs = F.log_softmax(traj_logits[0], dim=-1)
                token_lp = log_probs.gather(1, traj_tokens.unsqueeze(1)).squeeze(1)
                total_log_prob += token_lp[should_mask].sum().item()
                total_count += n_masked

        return total_log_prob / max(total_count, 1)

    # ── Geometric & hybrid scoring helpers for spec_dav ────────────

    def _parse_traj_from_tokens(self, traj_tokens):
        """Decode trajectory tokens → list of [x, y] waypoints."""
        import re
        text = self._tokenizer.decode(traj_tokens.tolist(), skip_special_tokens=True)
        pairs = re.findall(r'\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]', text)
        if not pairs:
            pairs = re.findall(r'([+-]?\d+\.\d+)\s*,\s*([+-]?\d+\.\d+)', text)
        if not pairs:
            return None
        return [[float(a), float(b)] for a, b in pairs]

    @staticmethod
    def _smoothness_score(waypoints):
        """Negative mean squared jerk — penalises jittery trajectories."""
        pts = np.array(waypoints, dtype=np.float64)
        if len(pts) < 4:
            return 0.0
        jerk = np.diff(pts, n=3, axis=0)
        return -float(np.mean(np.sum(jerk ** 2, axis=1)))

    @staticmethod
    def _physics_score(waypoints):
        """Soft penalty for implausible speed/acceleration (1 Hz, meters)."""
        pts = np.array(waypoints, dtype=np.float64)
        if len(pts) < 2:
            return 0.0
        vel = np.diff(pts, axis=0)
        speed = np.linalg.norm(vel, axis=1)
        penalty = float(np.sum(np.maximum(speed - 35.0, 0.0) ** 2))
        if len(pts) >= 3:
            acc = np.diff(vel, axis=0)
            acc_mag = np.linalg.norm(acc, axis=1)
            penalty += float(np.sum(np.maximum(acc_mag - 6.0, 0.0) ** 2)) * 2.0
        return -penalty

    @staticmethod
    def _consistency_score(waypoints, fmb_text):
        """Check trajectory consistency with future_meta_behavior text."""
        pts = np.array(waypoints, dtype=np.float64)
        if len(pts) < 2:
            return 0.0
        fmb = (fmb_text or "").lower()
        score = 0.0
        lat_disp = abs(pts[-1, 1] - pts[0, 1])
        if "straight" in fmb and lat_disp > 2.0:
            score -= (lat_disp - 2.0) * 0.5
        seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        if len(seg) >= 2:
            if ("decelerate" in fmb or "slow" in fmb) and seg[-1] > seg[0] * 1.3:
                score -= 1.0
            if "stop" in fmb and seg[-1] > 2.0:
                score -= 2.0
            if "accelerate" in fmb and seg[-1] < seg[0] * 0.7:
                score -= 1.0
        return score

    @staticmethod
    def _consensus_score(candidate_wps, all_candidates_wps):
        """Negative mean distance to centroid trajectory."""
        valid = [np.array(w, dtype=np.float64) for w in all_candidates_wps
                 if w is not None and len(w) >= 3]
        if len(valid) < 2:
            return 0.0
        min_len = min(len(v) for v in valid)
        valid = [v[:min_len] for v in valid]
        centroid = np.mean(valid, axis=0)
        cand = np.array(candidate_wps, dtype=np.float64)[:min_len]
        return -float(np.linalg.norm(cand - centroid, axis=1).mean())

    def _score_candidates(
        self, output_ids, candidates, traj_start, traj_end,
        scoring_mode="denoising", noise_levels=None, block_size=32,
        pixel_values=None, image_grid_thw=None,
    ):
        """Score trajectory candidates using the specified mode.

        Supported modes:
          - ``denoising``: denoising log-likelihood (requires forward passes)
          - ``geometric``: smoothness + physics + consensus (no model calls)
          - ``hybrid``:    weighted combination of denoising + geometric
          - ``consistency``: geometric + future_meta_behavior consistency
          - ``consensus_only``: pick the candidate closest to the centroid

        Returns (scores_list, details_dict).
        """
        N = len(candidates)
        noise_levels = noise_levels or [0.3, 0.6]

        # Parse waypoints for each candidate (needed by geometric modes)
        all_wps = [self._parse_traj_from_tokens(c) for c in candidates]

        # ── Denoising scores ───────────────────────────────────────
        if scoring_mode in ("denoising", "hybrid"):
            den_scores = []
            for traj_tokens in candidates:
                s = self._score_trajectory_denoising(
                    output_ids, traj_tokens, traj_start, traj_end,
                    noise_levels=noise_levels, block_size=block_size,
                    pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                )
                den_scores.append(s)
        else:
            den_scores = [0.0] * N

        # ── Geometric component scores ─────────────────────────────
        # Extract fmb_text once (used by consistency/hybrid modes)
        fmb_text = ""
        if scoring_mode in ("consistency", "hybrid"):
            full_text = self._tokenizer.decode(
                output_ids[0].tolist(), skip_special_tokens=True)
            fmb_idx = full_text.find("future_meta_behavior")
            if fmb_idx >= 0:
                traj_idx = full_text.find("trajectory", fmb_idx)
                fmb_text = full_text[fmb_idx:traj_idx] if traj_idx > fmb_idx else full_text[fmb_idx:fmb_idx+200]

        # Pre-compute consensus centroid once
        valid_wps = [np.array(w, dtype=np.float64) for w in all_wps
                     if w is not None and len(w) >= 3]
        if len(valid_wps) >= 2:
            _min_len = min(len(v) for v in valid_wps)
            _trimmed = [v[:_min_len] for v in valid_wps]
            _centroid = np.mean(_trimmed, axis=0)
        else:
            _centroid = None
            _min_len = 0

        geo_components = []
        for i, wps in enumerate(all_wps):
            if wps is None or len(wps) < 2:
                geo_components.append({
                    "smoothness": -10.0,
                    "physics": -10.0,
                    "consensus": -10.0,
                    "consistency": -10.0,
                })
            else:
                if _centroid is not None:
                    cand = np.array(wps, dtype=np.float64)[:_min_len]
                    consensus = -float(np.linalg.norm(cand - _centroid, axis=1).mean())
                else:
                    consensus = 0.0

                geo_components.append({
                    "smoothness": self._smoothness_score(wps),
                    "physics": self._physics_score(wps),
                    "consensus": consensus,
                    "consistency": self._consistency_score(wps, fmb_text),
                })

        # ── Combine scores per mode ────────────────────────────────
        if scoring_mode == "denoising":
            scores = den_scores

        elif scoring_mode == "geometric":
            # Z-normalize and weighted sum of geometric components
            weights = {"smoothness": 0.30, "physics": 0.25,
                       "consensus": 0.30, "consistency": 0.15}
            scores = self._hybrid_combine(geo_components, weights)

        elif scoring_mode == "hybrid":
            # Combine denoising + geometric
            weights = {"denoising": 0.40, "smoothness": 0.15, "physics": 0.15,
                       "consensus": 0.20, "consistency": 0.10}
            combined = []
            for i in range(N):
                d = dict(geo_components[i])
                d["denoising"] = den_scores[i]
                combined.append(d)
            scores = self._hybrid_combine(combined, weights)

        elif scoring_mode == "consistency":
            weights = {"smoothness": 0.25, "physics": 0.20,
                       "consensus": 0.25, "consistency": 0.30}
            scores = self._hybrid_combine(geo_components, weights)

        elif scoring_mode == "consensus_only":
            scores = [g["consensus"] for g in geo_components]

        else:
            scores = den_scores  # fallback

        details = {
            "den_scores": den_scores,
            "geo_components": geo_components,
            "all_wps_valid": [w is not None for w in all_wps],
        }
        return scores, details

    @staticmethod
    def _hybrid_combine(component_scores_list, weights):
        """Z-normalise each component across candidates, then weighted sum."""
        N = len(component_scores_list)
        if N == 0:
            return []
        components = list(weights.keys())
        raw = {c: np.array([s.get(c, 0.0) for s in component_scores_list],
                           dtype=np.float64) for c in components}
        normed = {}
        for c in components:
            std = raw[c].std()
            normed[c] = (raw[c] - raw[c].mean()) / std if std > 1e-8 else np.zeros(N)
        final = np.zeros(N)
        for c in components:
            final += weights.get(c, 0.0) * normed[c]
        return final.tolist()

    @staticmethod
    def _to_trajectory_only_question(question: str) -> str:
        """Keep the original input block but ask only for trajectory prediction."""
        traj_header = (
            "You are an expert autonomous driving agent.\n"
            "Task: Trajectory Prediction\n"
            "Predict the optimal 5-second future trajectory (5 waypoints, 1 s intervals).\n\n"
            "Input:"
        )
        input_marker = "Input:"
        if input_marker in question:
            return traj_header + question.split(input_marker, 1)[1]
        return traj_header + "\n" + question

    @torch.no_grad()
    def generate(
        self,
        image,
        question: str,
        max_new_tokens: int = 1024,
    ) -> Union[str, tuple]:
        """对单张或多张图 + 问题进行推理，返回原始文本输出。

        image: PIL.Image.Image 或 List[PIL.Image.Image]
        Returns str normally, or (str, dav_stats) when DaV mode is active.
        """
        if isinstance(image, list):
            images_list = image
        else:
            images_list = [image]

        _sv = getattr(self, '_section_version', None) or os.environ.get("SECTION_VERSION")
        if str(_sv) == "traj_only_spec":
            question = self._to_trajectory_only_question(question)

        content = []
        image_idx = 0
        if '<image>' in question:
            parts = question.split('<image>')
            for idx, part in enumerate(parts):
                if part:
                    content.append({"type": "text", "text": part})
                if idx < len(parts) - 1:
                    if idx == len(parts) - 2:
                        while image_idx < len(images_list):
                            content.append({"type": "image", "image": images_list[image_idx]})
                            image_idx += 1
                    else:
                        if image_idx < len(images_list):
                            content.append({"type": "image", "image": images_list[image_idx]})
                            image_idx += 1
        else:
            content.append({"type": "text", "text": question})
            while image_idx < len(images_list):
                content.append({"type": "image", "image": images_list[image_idx]})
                image_idx += 1

        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text],
            images=images_list,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        if self.is_mdm and self.use_mdm == 1:
            if hasattr(self.model, 'mdm_sample_deep_scaffold'):
                # Section-aware speculative decoding
                spec_kwargs = dict(
                    input_ids=inputs.input_ids,
                    tokenizer=self._tokenizer,
                    section_token_budgets=self.section_token_budgets,
                    max_tokens=max_new_tokens,
                    mask_id=self.mask_id,
                    threshold=self.confidence_threshold,
                    debug=False,
                    stream=False,
                )
                if self.sub_block_size is not None:
                    spec_kwargs["sub_block_size"] = self.sub_block_size
                if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                    spec_kwargs["pixel_values"] = inputs.pixel_values
                if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                    spec_kwargs["image_grid_thw"] = inputs.image_grid_thw

                _sv = os.environ.get("SECTION_VERSION", "scaffold_spec")
                if str(_sv) == "deep":
                    # Deep scaffold: pre-fill entire JSON structure, denoise values only
                    deep_kwargs = dict(
                        input_ids=inputs.input_ids,
                        tokenizer=self._tokenizer,
                            max_tokens=max_new_tokens,
                        mask_id=self.mask_id,
                        threshold=self.confidence_threshold,
                        block_size=self.block_size or 32,
                    )
                    if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                        deep_kwargs["pixel_values"] = inputs.pixel_values
                    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                        deep_kwargs["image_grid_thw"] = inputs.image_grid_thw
                    print(f"Using mdm_sample_deep_scaffold!!!!!!!!!!!!!!!!!!!!")
                    output_ids = self.model.mdm_sample_deep_scaffold(**deep_kwargs)
                elif str(_sv) == "scaffold_spec":
                    # Scaffold-guided self-speculative: scaffold structure is
                    # deterministic, value tokens generated via self-spec with KV cache
                    ss_kwargs = dict(
                        input_ids=inputs.input_ids,
                        tokenizer=self._tokenizer,
                        block_size=self.block_size or 32,
                        max_tokens=max_new_tokens,
                        mask_id=self.mask_id,
                        threshold=self.confidence_threshold,
                    )
                    if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                        ss_kwargs["pixel_values"] = inputs.pixel_values
                    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                        ss_kwargs["image_grid_thw"] = inputs.image_grid_thw
                    print(f"Using scaffold_speculative_sample!!!!!!!!!!!!!!!!!!!!")
                    output_ids = self.model.scaffold_speculative_sample(**ss_kwargs)
                elif str(_sv) == "scaffold_spec_ss_multi_traj":
                    # Shared-prefix multi-trajectory rollout: SS for sections 1-3
                    # at vt=0, then fork the KV cache N times and run SS on the
                    # trajectory section with verify_temperature > 0, finally
                    # equal-weight averaging the N trajectories.
                    ss_kwargs = dict(
                        input_ids=inputs.input_ids,
                        tokenizer=self._tokenizer,
                        block_size=self.block_size or 32,
                        max_tokens=max_new_tokens,
                        mask_id=self.mask_id,
                        threshold=self.confidence_threshold,
                    )
                    if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                        ss_kwargs["pixel_values"] = inputs.pixel_values
                    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                        ss_kwargs["image_grid_thw"] = inputs.image_grid_thw
                    ss_kwargs["num_traj_rollouts"] = int(os.environ.get("NUM_TRAJ_ROLLOUTS", "4"))
                    ss_kwargs["traj_verify_temperature"] = float(os.environ.get("TRAJ_VERIFY_TEMP", "0.5"))
                    ss_kwargs["traj_draft_temperature"] = float(os.environ.get("TRAJ_DRAFT_TEMP", "0.0"))
                    print(
                        f"Using scaffold_spec_with_ss_multi_traj "
                        f"(N={ss_kwargs['num_traj_rollouts']}, vt={ss_kwargs['traj_verify_temperature']})"
                        "!!!!!!!!!!!!!!!!!!!!"
                    )
                    output_ids = self.model.scaffold_spec_with_ss_multi_traj(**ss_kwargs)
                else:
                    # Default to deep scaffold for any unrecognized section version
                    deep_kwargs = dict(
                        input_ids=inputs.input_ids,
                        tokenizer=self._tokenizer,
                            max_tokens=max_new_tokens,
                        mask_id=self.mask_id,
                        threshold=self.confidence_threshold,
                        block_size=self.block_size or 32,
                    )
                    if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                        deep_kwargs["pixel_values"] = inputs.pixel_values
                    if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                        deep_kwargs["image_grid_thw"] = inputs.image_grid_thw
                    output_ids = self.model.mdm_sample_deep_scaffold(**deep_kwargs)
            else:
                # Check for spec_dav mode (works with any MDM checkpoint)
                _sv = getattr(self, '_section_version', None) or os.environ.get("SECTION_VERSION")

                common_kwargs = dict(
                    input_ids=inputs.input_ids,
                    tokenizer=self.processor.tokenizer,
                    block_size=self.block_size,
                    max_tokens=max_new_tokens,
                    mask_id=self.mask_id,
                    debug=False,
                )
                if self.sub_block_size is not None:
                    common_kwargs["sub_block_size"] = self.sub_block_size
                if hasattr(inputs, "pixel_values") and inputs.pixel_values is not None:
                    common_kwargs["pixel_values"] = inputs.pixel_values
                if hasattr(inputs, "image_grid_thw") and inputs.image_grid_thw is not None:
                    common_kwargs["image_grid_thw"] = inputs.image_grid_thw

                if str(_sv) == "ar":
                    common_kwargs["temperature"] = self.temperature
                    common_kwargs["top_p"] = self.top_p
                    common_kwargs["top_k"] = self.top_k
                    print(f"Using ar_sample_cache!!!!!!!!!!!!!!!!!!!!")
                    output_ids = self.model.ar_sample_cache(**common_kwargs)
                elif str(_sv) == "block_causal":
                    common_kwargs["threshold"] = self.confidence_threshold
                    print(f"Using block_causal_sample_cache!!!!!!!!!!!!!!!!!!!!")
                    output_ids = self.model.block_causal_sample_cache(**common_kwargs)
                else:
                    # Self-speculative decoding (standard or quadratic)
                    spec_kwargs = dict(common_kwargs)
                    spec_kwargs["stream"] = False
                    # quadratic 版本不接受 threshold 参数
                    if not self.use_quadratic:
                        spec_kwargs["threshold"] = self.confidence_threshold
                    print(f"Using speculative_block_causal_sample_cache!!!!!!!!!!!!!!!!!!!!")
                    output_ids = self.model.speculative_block_causal_sample_cache(**spec_kwargs)

                # ── spec_dav: re-rank trajectory via denoising likelihood ──
                if str(_sv) == "spec_dav":
                    prompt_len = inputs.input_ids.shape[1]
                    traj_start, traj_end = self._find_trajectory_token_range(
                        output_ids, prompt_len,
                    )

                    if traj_start is not None and traj_end is not None and traj_end > traj_start:
                        N = self._dav_num_candidates
                        temperatures = self._dav_temperatures_list
                        noise_levels = self._dav_noise_levels_list
                        block_size = self.block_size or 32

                        pv = getattr(inputs, "pixel_values", None)
                        igt = getattr(inputs, "image_grid_thw", None)

                        # Candidate 0 = greedy (already generated)
                        candidates = [output_ids[0, traj_start:traj_end].clone()]

                        # Candidates 1..N-1: re-denoise trajectory with temperature
                        for cand_idx in range(1, N):
                            temp = temperatures[cand_idx % len(temperatures)]
                            traj_tokens = self._redenoise_trajectory(
                                output_ids, traj_start, traj_end,
                                temperature=temp, block_size=block_size,
                                pixel_values=pv, image_grid_thw=igt,
                            )
                            candidates.append(traj_tokens)

                        # Score each candidate using the configured scoring mode
                        scoring_mode = getattr(self, '_dav_scoring_mode', 'denoising')
                        scores, score_details = self._score_candidates(
                            output_ids, candidates, traj_start, traj_end,
                            scoring_mode=scoring_mode,
                            noise_levels=noise_levels, block_size=block_size,
                            pixel_values=pv, image_grid_thw=igt,
                        )

                        # Select best with greedy margin
                        raw_best = max(range(N), key=lambda i: scores[i])
                        greedy_margin = self._dav_greedy_margin
                        if raw_best != 0 and greedy_margin > 0:
                            best_idx = 0 if (scores[raw_best] - scores[0]) < greedy_margin else raw_best
                        else:
                            best_idx = raw_best

                        if best_idx != 0:
                            output_ids = output_ids.clone()
                            output_ids[0, traj_start:traj_end] = candidates[best_idx]

                        self._last_dav_stats = {
                            "mode": f"spec_dav_{scoring_mode}",
                            "scores": scores,
                            "best_idx": best_idx,
                            "raw_best_idx": raw_best,
                            "temperatures": [temperatures[i % len(temperatures)] for i in range(N)],
                            "greedy_fallback": best_idx != raw_best,
                            "score_details": score_details,
                        }
        else:
            # 标准自回归
            gen_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": self.temperature > 0,
            }
            if self.temperature > 0:
                gen_kwargs["temperature"] = self.temperature
                gen_kwargs["top_p"] = self.top_p
                gen_kwargs["top_k"] = self.top_k

            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return response.strip()


# ══════════════════════════════════════════════════════════════════
#  单进程评估
# ══════════════════════════════════════════════════════════════════

def evaluate_samples(
    evaluator: WaymoEvaluatorSpeculative,
    samples: List[Dict],
    image_root: str,
    max_new_tokens: int,
    rank: int = 0,
) -> List[Dict]:
    results = []

    for item in tqdm(samples, desc=f"GPU{rank} 推理", position=rank, leave=True):
        sample_id = item.get("sample_id", "")
        image_field = item.get("image", "")
        gt_traj = item.get("future waypoints", [])
        if isinstance(gt_traj, str):
            try:
                gt_traj = json.loads(gt_traj)
            except Exception:
                gt_traj = []
        nav_cmd = item.get("navigation_command", "")
        conversations = item.get("conversations", [])

        question = ""
        gt_answer_template = ""
        for msg in conversations:
            if msg.get("from") == "human":
                question = msg.get("value", "")
            elif msg.get("from") == "assistant":
                gt_answer_template = msg.get("value", "")

        image_path = resolve_image_path(image_field, image_root)

        entry = {
            "sample_id":           sample_id,
            "image":               image_field,
            "image_full_path":     image_path,
            "navigation_command":  nav_cmd,
            "question":            question,
            "gt_answer_template":  gt_answer_template,
            "gt_trajectory":       gt_traj,
            "model_output_raw":    None,
            "model_output_parsed": None,
            "pred_trajectory":     None,
            "metrics":             {"ade": None, "fde": None, "valid": False},
            "error":               None,
        }

        if not question:
            entry["error"] = "missing question in conversations"
            results.append(entry)
            continue

        # Validate image path(s) exist
        if isinstance(image_path, list):
            missing = [p for p in image_path if not os.path.isfile(p)]
            if missing:
                entry["error"] = f"image not found: {missing}"
                results.append(entry)
                continue
        else:
            if not os.path.isfile(image_path):
                entry["error"] = f"image not found: {image_path}"
                results.append(entry)
                continue

        try:
            if isinstance(image_path, list):
                image = [Image.open(p).convert("RGB") for p in image_path]
            else:
                image = Image.open(image_path).convert("RGB")
            raw_output = evaluator.generate(image, question, max_new_tokens=max_new_tokens)
            entry["model_output_raw"] = raw_output
            entry["model_output_parsed"] = parse_full_output(raw_output)
            entry["pred_trajectory"] = parse_trajectory(raw_output)
            entry["metrics"] = compute_trajectory_metrics(entry["pred_trajectory"], gt_traj)
            # Capture DaV stats if available
            if hasattr(evaluator, '_last_dav_stats') and evaluator._last_dav_stats is not None:
                stats = evaluator._last_dav_stats
                # Convert to serializable format
                entry["dav_stats"] = {
                    k: (v.tolist() if hasattr(v, 'tolist') else v)
                    for k, v in stats.items()
                }
                evaluator._last_dav_stats = None
        except Exception as e:
            entry["error"] = traceback.format_exc()
            print(f"\n[GPU{rank}] 错误 sample_id={sample_id}: {e}", flush=True)

        results.append(entry)

    return results


# ══════════════════════════════════════════════════════════════════
#  多 GPU 并行
# ══════════════════════════════════════════════════════════════════

def worker_fn(
    rank: int,
    world_size: int,
    model_path: str,
    base_model: str,
    all_samples: List[Dict],
    image_root: str,
    max_new_tokens: int,
    min_pixels: Optional[int],
    max_pixels: Optional[int],
    return_dict: dict,
    block_size: Optional[int] = None,
    sub_block_size: Optional[int] = None,
    confidence_threshold: float = 0.0,
    use_mdm: int = 1,
    use_quadratic: int = 0,
    temperature: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 50,
    section_block_steps: Optional[dict] = None,
    section_token_budgets: Optional[dict] = None,
):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"
    shard = all_samples[rank::world_size]

    try:
        evaluator = WaymoEvaluatorSpeculative(
            model_path=model_path,
            base_model=base_model,
            device=device,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            block_size=block_size,
            sub_block_size=sub_block_size,
            confidence_threshold=confidence_threshold,
            use_mdm=use_mdm,
            use_quadratic=use_quadratic,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            section_block_steps=section_block_steps,
            section_token_budgets=section_token_budgets,
        )
        results = evaluate_samples(evaluator, shard, image_root, max_new_tokens, rank)
    except Exception as e:
        err_msg = traceback.format_exc()
        print(f"\n[GPU{rank}] Worker 崩溃: {e}\n{err_msg}", flush=True)
        results = []
        for item in shard:
            results.append({
                "sample_id":           item.get("sample_id", ""),
                "image":               item.get("image", ""),
                "navigation_command":  item.get("navigation_command", ""),
                "question":            "",
                "gt_answer_template":  "",
                "gt_trajectory":       item.get("future waypoints", []),
                "model_output_raw":    None,
                "model_output_parsed": None,
                "pred_trajectory":     None,
                "metrics":             {"ade": None, "fde": None, "valid": False},
                "error":               f"Worker {rank} crashed: {e}",
            })

    return_dict[rank] = results


# ══════════════════════════════════════════════════════════════════
#  聚合指标
# ══════════════════════════════════════════════════════════════════

def aggregate_metrics(predictions: List[Dict]) -> Dict[str, Any]:
    total    = len(predictions)
    failed   = sum(1 for p in predictions if p.get("error"))
    evaluated = total - failed

    valid_preds = [p for p in predictions if p["metrics"].get("valid")]
    n_valid = len(valid_preds)

    agg = {
        "total_samples":          total,
        "evaluated":              evaluated,
        "failed":                 failed,
        "valid_trajectory":       n_valid,
        "valid_trajectory_ratio": n_valid / evaluated if evaluated > 0 else 0.0,
    }

    if n_valid > 0:
        ades = [p["metrics"]["ade"] for p in valid_preds]
        fdes = [p["metrics"]["fde"] for p in valid_preds]
        agg["ade_mean"]   = float(np.mean(ades))
        agg["ade_std"]    = float(np.std(ades))
        agg["ade_median"] = float(np.median(ades))
        agg["fde_mean"]   = float(np.mean(fdes))
        agg["fde_std"]    = float(np.std(fdes))
        agg["fde_median"] = float(np.median(fdes))
    else:
        agg["ade_mean"] = agg["fde_mean"] = None

    return agg


# ══════════════════════════════════════════════════════════════════
#  入口
# ══════════════════════════════════════════════════════════════════

# Mode → (SECTION_VERSION env value, default confidence threshold).
_MODES = {
    "section_diffusion":  ("deep",                         0.9),  # Section Diffusion (SD): iterative MDM denoising over the scaffold.
    "scaffold_spec":      ("scaffold_spec",                0.0),  # Scaffold Spec (SS): scaffold-aware self-speculative decoding (paper canonical).
    "inference_scaling":  ("scaffold_spec_ss_multi_traj",  0.0),  # SS + multi-trajectory rollouts (test-time inference scaling).
}


def main():
    parser = argparse.ArgumentParser(
        description="Fast-dDrive — Waymo E2E open-loop evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_path", default="Efficient-Large-Model/Fast-dDrive",
                        help="Fast-dDrive checkpoint directory or HuggingFace id "
                             "(default: Efficient-Large-Model/Fast-dDrive paper checkpoint).")
    parser.add_argument("--eval_json", required=True,
                        help="Waymo E2E validation JSON.")
    parser.add_argument("--image_root", required=True,
                        help="Root directory that the image paths in --eval_json are relative to.")
    parser.add_argument("--output_dir", required=True,
                        help="Where to write predictions.json + metrics.json.")
    parser.add_argument("--mode", default="scaffold_spec", choices=sorted(_MODES),
                        help="Decoding path. scaffold_spec is the paper canonical (Scaffold Spec, SS).")
    parser.add_argument("--confidence_threshold", type=float, default=None,
                        help="Override the per-mode default threshold "
                             "(0.0 for scaffold_spec / inference_scaling, 0.9 for section_diffusion).")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to shard inference across.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Cap on number of samples (-1 = all).")
    args = parser.parse_args()

    section_version, default_threshold = _MODES[args.mode]
    confidence_threshold = args.confidence_threshold if args.confidence_threshold is not None else default_threshold
    os.environ["SECTION_VERSION"] = section_version

    # Frozen-by-release knobs (defaults from the paper).
    block_size = 32
    base_model = "Qwen/Qwen2.5-VL-3B-Instruct"
    min_pixels = max_pixels = 200704
    max_new_tokens = 1024

    project_root = os.getcwd()
    eval_json = args.eval_json if os.path.isabs(args.eval_json) else os.path.join(project_root, args.eval_json)
    image_root = args.image_root if os.path.isabs(args.image_root) else os.path.join(project_root, args.image_root)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading eval JSON: {eval_json}", flush=True)
    with open(eval_json, "r") as f:
        all_samples = json.load(f)
    if args.max_samples > 0:
        all_samples = all_samples[:args.max_samples]
    print(f"  {len(all_samples)} samples; mode={args.mode} (SECTION_VERSION={section_version}, "
          f"threshold={confidence_threshold})", flush=True)

    t0 = time.time()
    if args.num_gpus > 1:
        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        return_dict = manager.dict()
        processes = []
        for rank in range(args.num_gpus):
            p = mp.Process(
                target=worker_fn,
                args=(
                    rank, args.num_gpus, args.model_path, base_model,
                    all_samples, image_root, max_new_tokens,
                    min_pixels, max_pixels, return_dict,
                    block_size, None, confidence_threshold, 1, 0,
                    0.0, 1.0, 50, None, None,
                ),
            )
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

        shards = []
        for r in range(args.num_gpus):
            if r in return_dict:
                shards.append(return_dict[r])
            else:
                print(f"[warn] GPU{r} returned no results", flush=True)
                shards.append([])

        predictions = [None] * len(all_samples)
        for rank, shard in enumerate(shards):
            for local_i, result in enumerate(shard):
                global_i = rank + local_i * args.num_gpus
                if global_i < len(predictions):
                    predictions[global_i] = result
        predictions = [p for p in predictions if p is not None]
    else:
        evaluator = WaymoEvaluatorSpeculative(
            model_path=args.model_path,
            base_model=base_model,
            device="cuda",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            block_size=block_size,
            sub_block_size=None,
            confidence_threshold=confidence_threshold,
            use_mdm=1,
            use_quadratic=0,
            temperature=0.0,
            top_p=1.0,
            top_k=50,
            section_block_steps=None,
            section_token_budgets=None,
        )
        predictions = evaluate_samples(
            evaluator, all_samples, image_root, max_new_tokens, rank=0
        )

    elapsed = time.time() - t0
    print(f"\nInference done in {elapsed:.1f}s ({elapsed/max(len(predictions),1):.2f}s/sample)", flush=True)

    agg = aggregate_metrics(predictions)
    output = {
        "metadata": {
            "model_path": args.model_path,
            "eval_json": eval_json,
            "image_root": image_root,
            "timestamp": datetime.now().isoformat(),
            "num_gpus": args.num_gpus,
            "elapsed_s": round(elapsed, 2),
            "mode": args.mode,
            "section_version": section_version,
            "confidence_threshold": confidence_threshold,
            "block_size": block_size,
        },
        "aggregate_metrics": agg,
        "predictions": predictions,
    }

    predictions_path = os.path.join(output_dir, "predictions.json")
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(predictions_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": output["metadata"], "aggregate_metrics": agg}, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print(f"Fast-dDrive eval — mode={args.mode}")
    print("=" * 60)
    print(f"  samples           : {agg['total_samples']}")
    print(f"  succeeded         : {agg['evaluated']}")
    print(f"  failed            : {agg['failed']}")
    print(f"  trajectory parse  : {agg['valid_trajectory']} ({agg['valid_trajectory_ratio']*100:.1f}%)")
    if agg.get("ade_mean") is not None:
        print(f"  ADE  (mean / median): {agg['ade_mean']:.4f} / {agg['ade_median']:.4f} m")
        print(f"  FDE  (mean / median): {agg['fde_mean']:.4f} / {agg['fde_median']:.4f} m")
    print("=" * 60)
    print(f"Saved:\n  {predictions_path}\n  {metrics_path}")


if __name__ == "__main__":
    main()
