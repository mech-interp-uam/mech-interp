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

os.environ["OPENAI_API_KEY"] = getpass.getpass("Clave OpenAI: ")

raw_ds = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR)
raw_ds = raw_ds.with_format("torch", columns=["activations"])
print(f"   {len(raw_ds):,} ejemplos cargados")

map_location = "cuda" if torch.cuda.is_available() else "cpu"
print("Cargando checkpoint SAE…")
ckpt_path = hf_hub_download(REPO_ID, "sae.pth", cache_dir=CACHE_DIR)
state   = torch.load(ckpt_path, map_location=map_location)
W_DEC   = state["dec.weight"].to(DEVICE, DTYPE)   # (d_in, d_sae)
B_DEC   = state["dec.bias"  ].to(DEVICE, DTYPE)
TAU     = state["log_threshold"].exp().to(DEVICE, DTYPE)

D_IN  = W_DEC.shape[0]
D_SAE = W_DEC.shape[1]
W_ENC = W_DEC  
print(f"d_sae = {D_SAE:,}")

BIAS_TOTAL: Float[Tensor, "d_sae"] = - (B_DEC @ W_ENC) - TAU

D_SAE = W_ENC.shape[1]
print(f"   d_sae = {D_SAE:,}")

# ───── 3. Encoder funcional ───────────────────────────────────────────────── ───────────────────────────────────────────────────
@torch.no_grad()
def encode_batch(x: Float[Tensor, "b d_in"]) -> Float[Tensor, "b d_sae"]:
    return (x - B_DEC) @ W_ENC - TAU @ W_ENC - TAU



def collect_stats_and_topk(
    dataset: Iterable[Tensor],
    k: int = TOP_K,
    batch_size: int = BATCH_SIZE,
) -> tuple[Int[Array, "d_sae"], Int[Array, "d_sae k"], Float[Array, "d_sae k"]]:
    """Devuelve counts y los top‑k valores/índices por neurona."""
    val_heap = torch.full((D_SAE, TOP_K), -torch.inf, device=DEVICE, dtype=DTYPE)
    idx_heap = torch.full_like(val_heap, -1, dtype=torch.int64)
    counts   = torch.zeros(D_SAE, dtype=torch.int64)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=0,
                                         pin_memory=True)
    offset = 0
    for acts in tqdm(loader, desc="Recolectando top‑k", unit="batch"):
        acts = (acts["activations"] / EXP_NORM).to(DEVICE, DTYPE, non_blocking=True)
        lat  = encode_batch(acts)                                      # (B,d_sae)
#quitar el relu para optimizar la complejidad
# sumar los bias - combinarlos bias del encoder y decoder
        counts += (lat > 0).sum(0)

        combined_vals = torch.cat([val_heap, lat.T], dim=1)            # (d_sae, k+B)
        combined_idx  = torch.cat([
            idx_heap,
            torch.arange(offset, offset+lat.shape[0], device=DEVICE)
                  .repeat_interleave(D_SAE).view(D_SAE, -1)
        ], dim=1)
        vals, idxs = torch.topk(combined_vals, k, dim=1, largest=True)
        val_heap, top_pos = vals, idxs
        idx_heap = combined_idx.gather(1, top_pos)
        offset += lat.shape[0]

    return (counts.cpu().numpy(),
            idx_heap.cpu().numpy(),
            val_heap.cpu().numpy())

MODEL_NAME  = "gpt-4o-mini"
MAX_NEURONS = 100
TEMP        = 0.3


def create_prompt(nid: int, acts_idx: np.ndarray, acts_val: np.ndarray) -> str:
    pairs = "\n".join(f"• idx={i:7d}, act={v:>.3f}"
                       for i, v in zip(acts_idx, acts_val))
    return textwrap.dedent(f"""
        Eres un analista experto que interpreta neuronas latentes de un Sparse Auto‑Encoder.
        Analiza la neurona #{nid}.
        Estos son sus {len(acts_val)} ejemplos con activación más alta:

        {pairs}

        1. Hipótesis en español de qué concepto lingüístico, semántico y didáctico detecta.
        2. ¿Qué patrones observas en las activaciones?
        3. Da 2-3 frases‑ejemplo donde esperas que se active.
        4. Confianza (0‑100%).
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
    counts, idx_topk, val_topk = collect_stats_and_topk(raw_ds, TOP_K, BATCH_SIZE)

    live = np.count_nonzero(counts)
    print(f"Neuronas vivas: {live}/{D_SAE}  (μ activaciones = {counts.mean():.1f})")

    # seleccionar neuronas más activas
    active_ids = counts.argsort()[-MAX_NEURONS:][::-1]

    results = []
    for nid in tqdm(active_ids, desc="Explicando neuronas"):
        explanation = explain_neuron_robust(
            int(nid), idx_topk[nid], val_topk[nid])
        results.append({"neuron": int(nid), "explanation": explanation})
        print("\n" + "="*80 + f"\n Neurona {nid}\n" + explanation)

    with open("neuron_explanations.jsonl", "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\nExplicaciones guardadas en neuron_explanations.jsonl")