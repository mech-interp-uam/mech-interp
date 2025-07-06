#!/usr/bin/env python
# coding: utf-8

# # Interpretación de latentes 

# In[1]:


from pathlib import Path
from typing import Iterator, Tuple, Dict, List

import torch, numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download


# In[ ]:


dataset_ID   = "mech-interp-uam/llama-mlp8-outputs"  
repo_id  = "mech-interp-uam/llama3.2-1b-sae" 
CACHE_DIR    = Path("/home/joulesd/hf_cache")
batch_size   = 1024
top_k       = 20
act_thr      = 0.0
device       = "cuda" if torch.cuda.is_available() else "cpu"
dtype        = torch.float16  


# In[3]:


ds = load_dataset(
    dataset_ID,
    split="train",                    
    cache_dir=CACHE_DIR,
    streaming=False                   
).with_format("torch", columns=["activations"])


# In[ ]:


from sae import Sae                                    # o la ruta correcta

state_path = hf_hub_download(repo_id, "sae.pth")
state_dict = torch.load(state_path, map_location="cpu")
sae = Sae(d_in=2048, d_sae=2048*8, use_pre_enc_bias=True) \
        .to(device, dtype).eval()
_ = sae.load_state_dict(state_dict, strict=False)


# In[ ]:


def collect_stats_and_topk(
    dataset,
    sae_model: Sae,
    k: int = top_k,
    batch_size: int = batch_size,
    act_thr: float = act_thr,
) -> Tuple[np.ndarray,
           Dict[int, List[Tuple[float, int]]]]:
    d_sae   = sae_model.W_enc.shape[0]        
    counts  = np.zeros(d_sae, dtype=np.int64)
    heaps   = {i: [] for i in range(d_sae)} 

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    example_offset = 0
    with torch.no_grad():
        for batch in loader:
            acts = batch["activations"].to(device, dtype) 
            lat  = sae_model(acts)["latent"].cpu().numpy()     
            activated = lat > act_thr
            counts += activated.sum(axis=0)

            for row_idx, vec in enumerate(lat):
                global_idx = example_offset + row_idx
                for j, val in enumerate(vec):

                    if len(heaps[j]) < k:
                        heapq.heappush(heaps[j], (val, global_idx))
                    elif val > heaps[j][0][0]:
                        heapq.heapreplace(heaps[j], (val, global_idx))

            example_offset += lat.shape[0]

    topk = {j: sorted(h, reverse=True) for j, h in heaps.items()}
    return counts, topk

import heapq
counts, topk_by_neuron = collect_stats_and_topk(ds, sae)


# In[ ]:


def iter_topk_batches(
    dataset,
    topk_dict: Dict[int, List[Tuple[float, int]]],
    batch_size: int = batch_size,
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    samples = []
    for j, lst in topk_dict.items():
        for rank, (_, ex_idx) in enumerate(lst):
            samples.append((ex_idx, j, rank))
    samples.sort()


    unique_idx = sorted({ex for ex, _, _ in samples})
    id2acts = {i: dataset[i]["activations"] for i in unique_idx}

    acts_batch, lat_batch, meta_batch = [], [], []
    for ex_idx, neuron_id, rank in samples:
        vec = id2acts[ex_idx].unsqueeze(0).to(device, dtype)   
        with torch.no_grad():
            lat = sae(vec)["latent"]

        acts_batch.append(vec.squeeze(0).cpu())
        lat_batch .append(lat.squeeze(0).cpu())
        meta_batch.append(torch.tensor([neuron_id, rank]))

        if len(acts_batch) == batch_size:
            yield (torch.stack(acts_batch),
                   torch.stack(lat_batch),
                   torch.stack(meta_batch))
            acts_batch, lat_batch, meta_batch = [], [], []

    if acts_batch:
        yield (torch.stack(acts_batch),
               torch.stack(lat_batch),
               torch.stack(meta_batch))


# In[ ]:


import os, openai
openai.api_key = os.environ["OPENAI_API_KEY"]

def explain_neuron(neuron_id: int, topk_lat: torch.Tensor) -> str:
    prompt = f"""
    Eres un asistente que interpreta neuronas latentes de un SAE.
    La neurona #{neuron_id} tiene estas {len(topk_lat)} activaciones top-k:

    {topk_lat.tolist()}

    Explica en español qué concepto lingüístico o semántico capta
    esta neurona y da dos ejemplos de frases donde esperas que se active.
    """
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role":"user","content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# In[ ]:


for acts, lat, meta in iter_topk_batches(ds, topk_by_neuron, 64):
    for i in range(lat.shape[0]):
        nid   = int(meta[i,0])
        rank  = int(meta[i,1])
        lvec  = lat[i]
        if rank == 0:        
            explanation = explain_neuron(nid, lat[meta[:,0]==nid])
            print(f"Neurona {nid}:")
            print(explanation)


# In[ ]:


dead = np.where(counts == 0)[0]
print(f"Neuronas completamente inactivas: {len(dead)} / {len(counts)}")


# In[ ]:


topk_by_neuron = {j: lst for j, lst in topk_by_neuron.items() if counts[j] > 0}

