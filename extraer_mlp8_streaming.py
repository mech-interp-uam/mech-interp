# extraer_mlp8_streaming.py
# Versión optimizada con:
# • streaming sin caché local
# • secuencias largas (8192)
# • descarte de tokens especiales
# • shards Parquet directos al Hub

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, Features, Sequence, Array2D
from torch.utils.data import IterableDataset, DataLoader
import torch, numpy as np, itertools, gc
import os

HF_TOKEN    = "inserta token aqui con permiso write"
HF_TOKEN = os.environ.get("HF_TOKEN")
REPO_ID     = "naraca/mi-dataset-activaciones-llama3_2"
MODEL_NAME  = "meta-llama/Llama-3.2-1B"
SAMPLE_DOCS = 5_000            # textos a procesar
MAX_LEN     = 8_192            # longitud de contexto
BATCH_SIZE  = 2                # ajusta según VRAM
LAYER_IDX   = 8                # MLP-8
SHARD_SIZE  = 1_000            # ejemplos por shard

login(HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto",
    #token=HF_TOKEN, tambien puede ser token=None
)
model.eval()

mlp_out = None
def hook_mlp(_mod, _inp, output):
    global mlp_out
    mlp_out = output.detach().cpu()

model.model.layers[LAYER_IDX].mlp.register_forward_hook(hook_mlp)

stream_ds = load_dataset(
    "oscar",
    "unshuffled_deduplicated_es",
    split="train",
    streaming=True,
    download_mode="force_redownload",
    verification_mode="no_checks",
    trust_remote_code=True  
)

stream_ds = stream_ds.filter(lambda x: x["text"].strip() != "")
stream_ds = stream_ds.shuffle(buffer_size=50_000, seed=42)
stream_ds = itertools.islice(stream_ds, SAMPLE_DOCS)

class OscarStream(IterableDataset):
    def __iter__(self):
        for ex in stream_ds:
            tok = tokenizer(
                ex["text"],
                add_special_tokens=True,
                truncation=True,
                padding="max_length",
                max_length=MAX_LEN,
                return_tensors="pt",
            )
            yield {
                "input_ids": tok["input_ids"][0],
                "attention_mask": tok["attention_mask"][0],
            }

loader = DataLoader(OscarStream(), batch_size=BATCH_SIZE)

hidden_dim = model.config.intermediate_size
from datasets import Features, Sequence, Value
features = Features({
    "activacion": Sequence(
        feature=Sequence(Value("float32"))
    )
})
#extraccion y subida
buffer, shard_idx, n_proc = [], 0, 0

with torch.no_grad():
    for batch in loader:
        batch = {k: v.to(model.device) for k, v in batch.items()}
        mlp_out = None
        _ = model(**batch)

        for i in range(batch["input_ids"].shape[0]):
            acts = mlp_out[i]                                # (seq_len, hid)
            ids  = batch["input_ids"][i].cpu()

            mask = ~torch.isin(ids, torch.tensor(tokenizer.all_special_ids))
            acts_np = acts[mask].cpu().tolist()


            buffer.append({"activacion": acts_np})
            n_proc += 1

        if len(buffer) >= SHARD_SIZE:
            Dataset.from_list(buffer, features=features).push_to_hub(
                REPO_ID,
                split="train",
                private=True,
                max_shard_size="500MB",
            )
            print(f"Shard {shard_idx} subido — {n_proc} ejemplos")
            buffer.clear()
            shard_idx += 1
            gc.collect(); torch.cuda.empty_cache()

if buffer:
    Dataset.from_list(buffer, features=features).push_to_hub(
        REPO_ID,
        split="train",
        private=True,
        max_shard_size="500MB",
    )
    print(f"Shard final subido — {n_proc} ejemplos totales")

print("Proceso completo sin errores.")