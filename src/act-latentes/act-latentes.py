from __future__ import annotations

import os, heapq, json, math, textwrap
from pathlib import Path
from collections.abc import Iterable, Iterator

import numpy as np
import torch
from torch import Tensor
from jaxtyping import Float, Int, Array
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
from huggingface_hub import login
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import openai
import getpass
import os
from dotenv import load_dotenv

load_dotenv()


DATASET_ID   = "naraca/llama3.2-1b-sae"
REPO_ID      = "mech-interp-uam/llama3.2-1b-sae"
CACHE_DIR    = Path(os.getenv("HF_CACHE_DIR", "~/.cache/hf_cache")).expanduser()
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE        = torch.float16
BATCH_SIZE   = 1024
TOP_K        = 20
EXP_NORM     = 3.4

HF_TOKEN =
raw_ds = load_dataset(DATASET_ID, split="train", cache_dir=CACHE_DIR)

print(raw_ds.features)
raw_ds = raw_ds.rename_column("activacion", "activations")
raw_ds = raw_ds.rename_column("token_id", "input_ids") 
raw_ds = raw_ds.with_format("torch", columns = ["activations", "input_ids", "doc_id", "tok_pos"])
print(f"   {len(raw_ds):,} ejemplos cargados")
print(raw_ds.features)
map_location = "cuda" if torch.cuda.is_available() else "cpu"
print("Cargando checkpoint SAE…")
ckpt_path = hf_hub_download(REPO_ID, "sae_exp24_sparse0.001_d_sae_std_fullwarmup_steps256000_lr7e-05.pth", cache_dir=CACHE_DIR)
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
print(f"B_ENC shape: {B_ENC.shape}, W_ENC shape: {W_ENC.shape}  ")
BIAS_TOTAL: Float[Array, "d_sae"] = B_ENC - (B_DEC @ W_ENC.transpose(0, 1))

print(f"   d_sae = {D_SAE:,}")
login(token = HF_TOKEN, add_to_git_credential=True)
MODEL_NAME_FOR_TOKENIZER = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME_FOR_TOKENIZER, 
    cache_dir=CACHE_DIR,
)


doc_tokens_map = {}

print("Pre-procesando dataset para mapear doc_id a tokens (esto puede tardar y usar RAM)...")

temp_ds_iter = raw_ds.with_format(None)
for item in tqdm(temp_ds_iter, desc="Building doc-token map"):
    doc_id = item['doc_id']
    tok_pos = item['tok_pos']
    token_id = item['input_ids'] # It's a tensor here, convert to item() if not already done by with_format(None)
    if isinstance(token_id, torch.Tensor):
        token_id = token_id.item() # Convert to Python int if it's a tensor scalar

    if doc_id not in doc_tokens_map:
        doc_tokens_map[doc_id] = []
    
    doc_tokens_map[doc_id].append((tok_pos, token_id))

# Sort tokens within each document by tok_pos and extract only token_ids
for doc_id in doc_tokens_map:
    doc_tokens_map[doc_id].sort(key=lambda x: x[0]) # Sort by tok_pos
    doc_tokens_map[doc_id] = [x[1] for x in doc_tokens_map[doc_id]] # Keep only token_id

print(f"Mapa de {len(doc_tokens_map):,} documentos construido.")


@torch.no_grad()
def encode_batch(
    x: Float[Array, "b d_model"], # x es la entrada (activaciones MLP)
    W_enc_param: Float[Array, "d_sae d_model"], # Pesos del encoder
    bias_total_param: Float[Array, "d_sae"] # Sesgo total para el encoder
) -> Float[Array, "b d_sae"]:
    return x @ W_enc_param.transpose(0,1) + bias_total_param



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
            val_activ.cpu().numpy())

MODEL_NAME  = "gpt-4o-mini"
MAX_NEURONS = 100
TEMP        = 0.3

OPENAI_API_KEY =
openai.api_key = OPENAI_API_KEY
def create_prompt(nid: int, acts_idx: np.ndarray, acts_val: np.ndarray, raw_dataset: Dataset, tokenizer_param: AutoTokenizer, doc_map: dict) -> str:
    example_texts = []
    context_window_size = 5 
    for i, global_idx in enumerate(acts_idx):
        
        example_data = raw_dataset[int(global_idx)]
        
        doc_id = example_data['doc_id']
        tok_pos = example_data['tok_pos']
        token_id_at_pos = example_data['input_ids'].item()

        context_tokens_ids_list = []
        full_doc_tokens = doc_map.get(doc_id)


        if full_doc_tokens:
            if 0 <= tok_pos < len(full_doc_tokens):
                start_context_idx = max(0, tok_pos - context_window_size)
                end_context_idx = min(len(full_doc_tokens), tok_pos + context_window_size + 1)
                context_tokens_ids_list = full_doc_tokens[start_context_idx:end_context_idx]

        context_text = tokenizer_param.decode(context_tokens_ids_list, skip_special_tokens=True)

        activated_token_text = tokenizer_param.decode([token_id_at_pos], skip_special_tokens=True)
        if activated_token_text in context_text:
             context_text = context_text.replace(activated_token_text, f"<<{activated_token_text}>>", 1)


        example_texts.append(
            f"• idx={global_idx:7d}, act={acts_val[i]:>.3f}, doc_id={doc_id}, tok_pos={tok_pos}, token_id={token_id_at_pos}\n"
            f"  Context: \"{context_text}\""
        )

    pairs = "\n".join(example_texts)
    full_prompt_for_ai = textwrap.dedent(f"""
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
    return full_prompt_for_ai, pairs


def explain_neuron_robust(nid: int, idx: np.ndarray, val: np.ndarray, raw_dataset: Dataset, tokenizer_param: AutoTokenizer, doc_map: dict) -> str:
    full_prompt_for_ai, examples_context_string = create_prompt(nid, idx, val, raw_dataset, tokenizer_param, doc_map)
    ai_response_content = "[Error inesperado en OpenAI]"
    if not openai.api_key:
        return {
            "ai_explanation": ai_response_content,
            "top_examples_context": examples_context_string # Pass the examples_context_string here
        }
    
    for attempt in range(3):
        try:
            rsp = openai.chat.completions.create(
                model=MODEL_NAME,
                temperature=TEMP,
                messages=[{"role": "user", "content": full_prompt_for_ai}],
            )
            ai_response_content = rsp.choices[0].message.content.strip()
            break
        except Exception as e:
            if attempt == 2:
                return f"[Error OpenAI: {e}]"
            torch.cuda.empty_cache()
    return {
        "ai_explanation": ai_response_content,
        "top_examples_context": examples_context_string
    }
    

if __name__ == "__main__":
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


    counts, idx_topk, val_topk = collect_stats_and_topk(raw_ds, TOP_K, BATCH_SIZE)

    live = np.count_nonzero(counts)
    print(f"Neuronas vivas: {live}/{D_SAE}  (μ activaciones = {counts.mean():.1f})")

    # seleccionar neuronas más activas
    active_ids = counts.argsort()[-MAX_NEURONS:][::-1]

    results = []
    for nid in tqdm(active_ids, desc="Explicando neuronas"):
        explanation_data = explain_neuron_robust(
            int(nid), idx_topk[:, int(nid)], val_topk[:, int(nid)], raw_ds, tokenizer, doc_tokens_map)
        results.append({
            "neuron": int(nid), 
            "top_examples_context": explanation_data["ai_explanation"],
            "ai_explanation": explanation_data["ai_explanation"]
            })
        print("\n" + "="*80 + f"\n Neurona {nid}\n")
        print("--- Explicación de la IA ---")
        print(explanation_data["ai_explanation"]) # Access AI explanation
        print("\n--- Contextos de alta activación (con token resaltado) ---")
        print(explanation_data["top_examples_context"])

    with open("neuron_explanations.jsonl", "w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("\nExplicaciones guardadas en neuron_explanations.jsonl")


