from __future__ import annotations

import os, heapq, json, math, textwrap
from pathlib import Path
from collections.abc import Iterable, Iterator

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float, Int, Array
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import openai
import getpass
import os

DATASET_ID   = "mech-interp-uam/llama-mlp8-outputs"
REPO_ID      = "mech-interp-uam/llama3.2-1b-sae"
CACHE_DIR    = Path(os.getenv("HF_CACHE_DIR", "~/.cache/hf_cache")).expanduser()
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.float16
BATCH_SIZE   = 4096
TOP_K        = 20
EXP_NORM     = 3.4



raw_ds = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR)
raw_ds = raw_ds.with_format("torch", columns=["activations"])
print(f"   {len(raw_ds):,} ejemplos cargados")

map_location = "cuda" if torch.cuda.is_available() else "cpu"
ckpt_path = hf_hub_download(REPO_ID, "sae.pth", cache_dir=CACHE_DIR)
state   = torch.load(ckpt_path, map_location=map_location)
W_ENC: Float[Array, "d_sae d_model"] = state["enc.weight"].to(DEVICE, DTYPE)
B_ENC: Float[Array, "d_sae"]         = state["enc.bias" ].to(DEVICE, DTYPE)
W_DEC: Float[Array, "d_model d_sae"] = state["dec.weight"].to(DEVICE, DTYPE)
B_DEC: Float[Array, "d_model"]       = state["dec.bias" ].to(DEVICE, DTYPE)
TAU:   Float[Array, "d_sae"]         = state["log_threshold"].exp().to(DEVICE, DTYPE)
D_MODEL = W_ENC.shape[1]
D_SAE   = W_DEC.shape[1] 
print(D_SAE, D_MODEL)
print(f"   d_sae = {D_SAE:,}")

BIAS_TOTAL: Float[Array, "d_sae"] = B_ENC - (B_DEC @ W_DEC)

@torch.no_grad()
def encode_batch(
    x: Float[Array, "b d_model"], # x es la entrada (activaciones MLP)
    W_enc_param: Float[Array, "d_sae d_model"], # Pesos del encoder
    bias_total_param: Float[Array, "d_sae"] # Sesgo total para el encoder
) -> Float[Array, "b d_sae"]:
    return x @ W_ENC.transpose(0,1) + BIAS_TOTAL  # (B, d_sae)


def collect_stats_and_topk(
    dataset: Iterable[Float[Array, "b d_model"]],
    k: int = TOP_K,
    batch_size: int = BATCH_SIZE,
    W_ENC_param: Float[Array, "d_sae d_model"] = W_ENC,
    BIAS_TOTAL_param: Float[Array, "d_sae"] = BIAS_TOTAL,
    TAU_param: Float[Array, "d_sae"] = TAU,
    exp_norm_param: float = EXP_NORM
) -> tuple[Int[Array, "d_sae"], Int[Array, "d_sae k"], Float[Array, "d_sae k"]]:
    """
    Collects activation counts and top-k values/indices per neuron.
    """
    val_activ = torch.full((k, D_SAE), -torch.inf, device=DEVICE, dtype=DTYPE)
    idx_activ = torch.full_like(val_activ, -1, dtype=torch.int64)

    counts   = torch.zeros(D_SAE, dtype=torch.int64)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0,
                                         pin_memory=True)
    offset = 0
    for acts_batch in tqdm(loader, desc="Collecting top-k", unit="batch"):
        acts = (acts_batch["activations"] / exp_norm_param).to(DEVICE, DTYPE, non_blocking=True)
        lat  = encode_batch(acts, W_ENC_param, BIAS_TOTAL_param) # lat shape: (B, D_SAE)

        current_batch_size = lat.shape[0]

        counts += (lat > TAU_param).sum(0)

        combined_vals = torch.cat([val_activ, lat], dim=0)

        batch_indices = torch.arange(offset, offset + current_batch_size, device=DEVICE)
        batch_indices_expanded = batch_indices.unsqueeze(1).repeat(1, D_SAE) # (B, 1) -> (B, D_SAE)

        combined_idx = torch.cat([idx_activ, batch_indices_expanded], dim=0)

        vals, idxs = torch.topk(combined_vals, k, dim=0, largest=True) # dim=0 instead of dim=1

        val_activ = vals
        idx_activ = combined_idx.gather(0, idxs)

        offset += current_batch_size

    return (counts.cpu().numpy(),
            idx_activ.cpu().numpy(), # Shape will be (k, D_SAE)
            val_activ.cpu().numpy()) # Shape will be (k, D_SAE)

MODEL_NAME  = "gpt-4o-mini"
MAX_NEURONS = 100
TEMP        = 0.3

os.environ["OPENAI_API_KEY"] = getpass.getpass("Pega tu clave: ")
def create_prompt(nid: int, acts_idx: np.ndarray, acts_val: np.ndarray) -> str:
    pairs = "\n".join(f"• idx={i:7d}, act={v:>.3f}"
                       for i, v in zip(acts_idx, acts_val))
    return textwrap.dedent(f"""
        You are an expert analyst specialized in interpreting latent neurons of a Sparse Auto-Encoders
        Your task is to anlyze neuron #{nid}.
        Here are its {len(acts_val)} highest activating examples:
        {pairs}

    Based on these examples, 

    1. **Hypothesis**: What concept, pattern, or feature does this neuron seem to detect? Be as specific as possible. 
    Consider any domain-specific knowledge that might help you interpret this neuron.
    2. **Observerd PAtterns**: Describe any commonalities or recurring themes you observe across the high-activating examples. What specific elements, contexts, or characteristics consistently lead to high activation?
    3. **Illustrative Examples**: Provide 2-3 hypothetical phrases or sentences where you would expect this neuron to activate strongly.
    4. **Confidence**: On a scale of 0-100%, how confident are you in your hypothesis and observations?
    """)


def explain_neuron_robust(nid: int, idx: np.ndarray, val: np.ndarray) -> str:
    if not openai.api_key:
        return "[OPENAI_API_KEY ausente]"

    prompt = create_prompt(nid, idx, val)
    for attempt in range(3):
        try:
            rsp = openai.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMP,
                messages=[{"role": "user", "content": prompt}],
            )
            return rsp.choices[0].message.content.strip()
        except Exception as e:
            if attempt == 2:
                return f"[Error OpenAI: {e}]"
            torch.cuda.empty_cache()
    return "[Error inesperado]"

if __name__ == "__main__":
    """
    print("\n--- Validating Encoder with L0 Norm-----")
    temp_loader = torch.utils.data.DataLoader(raw_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    try:
        sample_batch = next(iter(temp_loader))
    except StopIteration:
        print("Error: Dataset is empty or DataLoader failed to yield a batch. Exiting.")
        exit()

    acts_for_validation = (sample_batch["activations"] / EXP_NORM).to(DEVICE, DTYPE, non_blocking=True)
    sparse_activations = encode_batch(acts_for_validation, W_ENC, BIAS_TOTAL)

    l0_norm_per_example_tau = (sparse_activations > TAU).sum(dim=1)
    average_l0_tau = l0_norm_per_example_tau.float().mean().item()

    print(f"Average L0 Norm (activations > TAU): {average_l0_tau:.2f}")

    expected_min = 190
    expected_max = 230

    if expected_min <= average_l0_tau <= expected_max:
        print(f"Average L0 Norm ({average_l0_tau:.2f}) is within the expected range [{expected_min}-{expected_max}]! Your encoder works correctly.")
    else:
        print(f"Average L0 Norm ({average_l0_tau:.2f}) is OUTSIDE the expected range [{expected_min}-{expected_max}]. There might still be an issue.")
    """

    counts, idx_topk, val_topk = collect_stats_and_topk(raw_ds, TOP_K, BATCH_SIZE)

    live = np.count_nonzero(counts)
    print(f"Neuronas vivas: {live}/{D_SAE}  (μ activaciones = {counts.mean():.1f})")

    # seleccionar neuronas más activas
    active_ids = counts.argsort()[-MAX_NEURONS:][::-1]

    results = []
    for nid in tqdm(active_ids, desc="Explicando neuronas"):
        explanation = explain_neuron_robust(
            int(nid), idx_topk[:, int(nid)], val_topk[:, int(nid)])
        results.append({"neuron": int(nid), "explanation": explanation})
        print("\n" + "="*80 + f"\n Neuron {nid}\n" + explanation)

    with open("neuron_explanations.jsonl", "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\nExplicaciones guardadas en neuron_explanations.jsonl")