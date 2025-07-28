import os, gc, math, json, random, tempfile
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from datasets import load_dataset, Dataset, Features, Value, Sequence
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, HfApi
s
HF_TOKEN   = 
REPO_ID    = "naraca/activaciones-llama3-mlp8"
MODEL_NAME = "meta-llama/Llama-3.2-1B"

SAMPLE_DOCS = 100_000
SEQ_LEN     = 4_096
TRUNC_TOKS  = SEQ_LEN * 4
BATCH_FWD   = 16       
SHARD_SIZE  = 50_000

USE_FP16 = False
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

os.environ["HF_HOME"] = "/workspace/hf_cache"
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

login(token=HF_TOKEN, add_to_git_credential=True)

dtype = torch.float16 if USE_FP16 or not torch.cuda.is_bf16_supported() else torch.bfloat16
print(f" dtype: {dtype} — device: {DEVICE}")

tok = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=dtype if DEVICE == "cuda" else torch.float32,
    device_map="auto" if DEVICE == "cuda" else None,
).eval()

# hook MLP‑8
mlp_out = None
def hook(_m, _i, o):
    global mlp_out
    mlp_out = o.detach()
model.model.layers[8].mlp.register_forward_hook(hook)
special_ids = torch.tensor(tok.all_special_ids, device=DEVICE)

features = Features({
    "doc_id":    Value("int32"),
    "tok_pos":   Value("int32"),
    "token_id":  Value("int32"),
    "activacion": Sequence(Value("uint16")),
})

def good(t): t=t.strip(); return 20<len(t)<30_000 and t.upper()!=t and not t.isnumeric()

def chunker(text, did):
    ids = tok(text, add_special_tokens=False,
              truncation=True, max_length=TRUNC_TOKS).input_ids
    seq = [tok.bos_token_id] + ids + [tok.eos_token_id]
    for s in range(0, len(seq) - SEQ_LEN + 1, SEQ_LEN):
        yield seq[s:s+SEQ_LEN], did, s

stream = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True).take(SAMPLE_DOCS)

mini, doc_buf, pos_buf, push_buf = [], [], [], []
shard, sum_sq, n_tok = 0, 0.0, 0  

for did, ex in enumerate(stream):
    if not good(ex["text"]): continue
    for chunk, d, p in chunker(ex["text"], did):
        mini.append(chunk); doc_buf.append(d); pos_buf.append(p)
        if len(mini) == BATCH_FWD:
            gc.collect(); torch.cuda.empty_cache()
            ids = torch.tensor(mini, dtype=torch.long, device=DEVICE)[:, :1024]  # CORTADO
            with torch.no_grad():
                _ = model.model(input_ids=ids, attention_mask=torch.ones_like(ids))
            gc.collect(); torch.cuda.empty_cache()
            acts, mask = mlp_out, ~torch.isin(ids, special_ids)
            for i in range(len(mini)):
                ids_k  = ids[i][mask[i]].cpu()
                acts_k = acts[i][mask[i]].to(dtype).view(torch.uint16).cpu()
                sum_sq += acts_k.to(torch.float32).pow(2).sum(); n_tok += len(acts_k)
                for j, (tid, bits) in enumerate(zip(ids_k, acts_k)):
                    push_buf.append({"doc_id": doc_buf[i],"tok_pos": pos_buf[i]+j,
                                     "token_id": int(tid),"activacion": bits.numpy()})
            mini.clear(); doc_buf.clear(); pos_buf.clear()

            if len(push_buf) >= SHARD_SIZE:
                split = f"train_{shard:05d}"
                try:
                    Dataset.from_list(push_buf, features=features).push_to_hub(
                        REPO_ID, split=split, private=True, max_shard_size="500MB")
                    print(f" {len(push_buf)} filas — {split}")
                except Exception as e:
                    print(f" Falló la subida de {split} — se guarda localmente.\n{e}")
                    Dataset.from_list(push_buf, features=features).save_to_disk(f"/workspace/fallback_{split}")
                push_buf.clear(); shard += 1
                gc.collect(); torch.cuda.empty_cache()

# Final: procesa los restos
if mini:
    gc.collect(); torch.cuda.empty_cache()
    ids = torch.tensor(mini, dtype=torch.long, device=DEVICE)[:, :1024]
    with torch.no_grad():
        _ = model.model(input_ids=ids, attention_mask=torch.ones_like(ids))
    gc.collect(); torch.cuda.empty_cache()
    acts, mask = mlp_out, ~torch.isin(ids, special_ids)
    for i in range(len(mini)):
        ids_k  = ids[i][mask[i]].cpu()
        acts_k = acts[i][mask[i]].to(dtype).view(torch.uint16).cpu()
        sum_sq += acts_k.to(torch.float32).pow(2).sum(); n_tok += len(acts_k)
        for j, (tid, bits) in enumerate(zip(ids_k, acts_k)):
            push_buf.append({"doc_id": doc_buf[i],"tok_pos": pos_buf[i]+j,
                             "token_id": int(tid),"activacion": bits.numpy()})

if push_buf:
    split = f"train_{shard:05d}"
    try:
        Dataset.from_list(push_buf, features=features).push_to_hub(
            REPO_ID, split=split, private=True, max_shard_size="500MB")
        print(f" shard final ({len(push_buf)}) — {split}")
    except Exception as e:
        print(f" Falló la subida del shard final {split} — guardado localmente.\n{e}")
        Dataset.from_list(push_buf, features=features).save_to_disk(f"/workspace/fallback_{split}")

rms = math.sqrt(sum_sq / n_tok)
with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as f:
    json.dump({"tokens": n_tok, "rms": rms}, f, indent=2)
    stats_path = f.name
HfApi().upload_file(
    path_or_fileobj=stats_path,
    path_in_repo="norm_stats.json",
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN
)

print(f" Fin — dtype {dtype}, RMS = {rms:.4f}")