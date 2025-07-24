#!/usr/bin/env python
# extraer_mlp8_streaming.py  – listo para entrenar SAE (GemmaScope + RMS)

from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, Features, Sequence, Value
from torch.utils.data import IterableDataset, DataLoader
import torch, itertools, gc, os, json, tempfile, math

HF_TOKEN   =   
REPO_ID = "naraca/llama3.2-1b-sae" 

MODEL_NAME = "meta-llama/Llama-3.2-1B"
SAMPLE_DOCS = 200 # cambiar a 100_000
MAX_LEN     = 4096
BATCH_SIZE  = 1
LAYER_IDX   = 8
SHARD_SIZE  = 50

login(token=HF_TOKEN)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
eos_id = tokenizer.eos_token_id
bos_id = tokenizer.bos_token_id
#modelo 
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16, device_map="auto"
) 
model.eval() # para no usar gradientes
# quantization_config=quant en lugar de torch_dtype
#from transformers import BitsAndBytesConfig
#quant = BitsAndBytesConfig(load_in_4bit=True,
                           #bnb_4bit_use_double_quant=True)
#model = AutoModelForCausalLM.from_pretrained(
    #MODEL_NAME, quantization_config=quant, device_map="auto")

# Hook MLP-8
mlp_out = None
def hook_mlp(_m, _i, out):
    global mlp_out
    mlp_out = out.detach()
model.model.layers[LAYER_IDX].mlp.register_forward_hook(hook_mlp) #metemos el hohok a la capa 

# valid_sources = {"Wikipedia (en)", "StackExchange", "PubMed Abstracts", "ArXiv", }
                 
raw_ds = load_dataset(
    "monology/pile-uncopyrighted",
    split="train",
    streaming=True,
    trust_remote_code=True,
    download_mode="reuse_dataset_if_exists",  # o force_redownload si prefieres
    
).filter(lambda x: (
    (txt := x["text"].strip()) != "" and
    len(txt) > 20 and
    sum(c.isalpha() for c in txt) >= 2 and
    txt.upper() != txt and
    not txt.isnumeric()
)).map(
    lambda ex, idx: {"index": idx, "text": ex["text"]},
    with_indices=True
).shuffle(buffer_size=50_000, seed=42) # enumeramos para producir {index: int, text: str}

eos = tokenizer.eos_token_id
bos = tokenizer.bos_token_id

# raw_ds = raw_ds.batch(8190).map(lambda x: [[[eos] + y + [bos] for y in tokenizer(x["text"], add_special_tokens=False)]])
# raw_ds = itertools.islice(raw_ds, SAMPLE_DOCS)

def tokenize_batch(batch):
    # batch["index"] y batch["text"] vienen de .enumerate()
    tokenized = tokenizer(
        batch["text"],
        add_special_tokens=False
    ).input_ids

    docs, toks, pos = [], [], []
    for doc_id, seq in zip(batch["index"], tokenized):
        seq = [bos_id] + seq + [eos_id]
        docs.extend([doc_id] * len(seq))
        toks.extend(seq)
        pos.extend(list(range(len(seq))))  # posiciones dentro del doc

    return {
        "doc_id": docs,
        "token_id": toks,
        "tok_pos": pos,
    }
raw_ds = raw_ds.map(tokenize_batch,batched=True, batch_size=1000,remove_columns=["index", "text"],)
for row in raw_ds.take(3):
    print(row)

def token_stream():
    for ex in raw_ds:
        for tok in ex["text"]:      # ex["text"] es List[int]
            yield tok
    #for ex in raw_ds:
        #yield from tokenizer(ex["text"], add_special_tokens=False).input_ids
# alternative: ds.map explode dos veces, seguido por .batched(batch_llama*seqlen) .map(convert to torch y reshape )
def token_stream2(raw_ds):
    iterador = iter(raw_ds)                # convertimos d.s a iterador
    tokens_por_batch = MAX_LEN * BATCH_SIZE # numero total de tokens por lote
    leftover: list[int] = []         # tokens que sobran de la iteracion pasada
    while True:   # buble infinito para generar bloques continuamente
        l = []          # lista que contendrá las secuencias de tokens del batch
        tokens_current_batch = 0 # contador del total de tokens acumulados en este batch

        if leftover:           # si hay tokens sobrantes del ciclo anterior...
            l.append(leftover)    # ...los incluimos al inicio del nuevo batch
            tokens_current_batch += len(leftover)  # actualizamos cuántos tokens ya llevamos
            leftover = []                 # limpiamos leftover después de usarlo
        try:  # intentamos llenar el bathc con nuevos ejemplos
            while tokens_current_batch < tokens_por_batch: #
                ejemplo = next(iterador)            # {'text': [bos, ..., eos]}
                tokens = ejemplo["text"]        # extraemos la lista de tokens ya tokenizada
                len_tokens = len(tokens) # contamos cuantos tokens contine

                if tokens_current_batch + len_tokens <= tokens_por_batch:
                    l.append(tokens)  # si cabe completo lo agregamos entero
                    tokens_current_batch += len_tokens
                else:
                    espacio_restante = tokens_por_batch - tokens_current_batch # tokens libres que quedan

                    if espacio_restante > 0:
                        l.append(tokens[:espacio_restante]) # agregamos solo lo que cabe
                        tokens_current_batch += espacio_restante
                    leftover = tokens[espacio_restante:] # guardamos el resto para el sigueinte lote
                    break  # batch lleno salimos del bluce interno
        except StopIteration:                       # se acabó el dataset
            if l:                           # si hay tokens aun 
                yield l     # emite lo último pendiente
            break                                   # sale del while True

        
        if l: # si el batch actual esta lleno se emite el lote al consumidro 
            yield l

class PackedPile(IterableDataset):
    def __iter__(self):
        buf_ids, buf_doc, buf_pos = [], [], []                               # donde se van pegando tokens
        for ex in raw_ds:             #     token_stream emite UN token a la vez
            buf_ids.append(ex["token_id"])
            buf_doc.append(ex["doc_id"])
            buf_pos.append(ex["tok_pos"])
                               # añade el token al final del buffer
            while len(buf_ids) >= MAX_LEN:
                yield self._make_item(
                    buf_ids[:MAX_LEN],
                    buf_doc[:MAX_LEN],
                    buf_pos[:MAX_LEN],
                )
                # descartamos lo consumido
                buf_ids  = buf_ids [MAX_LEN:]
                buf_doc  = buf_doc [MAX_LEN:]
                buf_pos  = buf_pos [MAX_LEN:]            # deja en buf lo que sobró
       
    def _make_item(self, ids, doc, pos):
        ids = torch.tensor(ids, dtype=torch.long)
        return {
            "input_ids":     ids,                       # (S,)
            "attention_mask": torch.ones_like(ids),     # (S,)
            "doc_id":        torch.tensor(doc),         # (S,)
            "tok_pos":       torch.tensor(pos),         # (S,)
        }        

loader = DataLoader(PackedPile(), batch_size=BATCH_SIZE, num_workers=0)

special_ids = torch.tensor(tokenizer.all_special_ids, device=model.device)

features = Features({
    "doc_id":    Value("int32"),
    "tok_pos":   Value("int32"),
    "token_id":  Value("int32"),
    "activacion": Sequence(feature=Value("float16")),
})#

import random
buffer, shard_idx, n_proc = [], 0, 0
sum_sq, n_tokens = 0.0, 0.0          # para RMS

with torch.no_grad(): 
    for batch in loader: # toma batch del DL cada bthc dic inputo_ids y attention_mask (b_s, m_l)
        inp  = batch["input_ids"].to(model.device)
        attn = batch["attention_mask"].to(model.device)
        doc  = batch["doc_id"]    .to(model.device)    # (B,S)
        pos  = batch["tok_pos"]   .to(model.device)
        _ = model(input_ids=inp, attention_mask=attn)
        # aqyu batch tiene que ser matrix cuadrada de enteros
        #_ = model(batch)

        for i in range(inp.size(0)):                   # por cada secuencia
            ids_row  = inp[i]
            acts_row = mlp_out[i]          # (S, hidden)
            doc_row  = doc[i]
            pos_row  = pos[i]

            mask = ~torch.isin(ids_row, special_ids)   # True = token normal
            print(f"acts_row.device = {acts_row.device}, mask.device = {mask.device}")
            ids_keep  = ids_row [mask]
            acts_keep = acts_row[mask]
            doc_keep  = doc_row[mask]
            pos_keep  = pos_row[mask]

            ids_keep  = ids_keep.cpu()
            acts_keep = acts_keep.cpu()
            doc_keep  = doc_keep.cpu()
            pos_keep  = pos_keep.cpu()

            # RMS global
            sum_sq   += acts_keep.float().pow(2).sum().item()
            n_tokens += acts_keep.shape[0]

            # Empaquetamos ejemplo por ejemplo
            for d, p, t, vec in zip(doc_keep, pos_keep, ids_keep, acts_keep):
                buffer.append({
                    "doc_id": int(d),
                    "tok_pos": int(p),
                    "token_id": int(t),
                    "activacion": vec.half().numpy(),  # list[float16]
                })

        # Subida por lotes
        if len(buffer) >= SHARD_SIZE:
            random.shuffle(buffer)
            Dataset.from_list(buffer, features=features).push_to_hub(
                REPO_ID, split="train", private=True,
                max_shard_size="500MB"
            )
            print(f"Shard {shard_idx} subido — {len(buffer)} filas")
            buffer.clear(); shard_idx += 1
            gc.collect(); torch.cuda.empty_cache()

if buffer:
    random.shuffle(buffer)
    Dataset.from_list(buffer, features=features)\
           .push_to_hub(REPO_ID, split="train", private=True,
                        max_shard_size="500MB")
    print(f"Shard final subido con {len(buffer)} ejemplos — {n_proc} ejemplos totales")



rms = math.sqrt(sum_sq / n_tokens)
stats = {"tokens": n_tokens, "rms": rms}
with tempfile.NamedTemporaryFile("w", delete=False, suffix=".json") as tmp:
    json.dump(stats, tmp, indent=2)
    stats_path = tmp.name

HfApi().upload_file(
    path_or_fileobj=stats_path,
    path_in_repo="norm_stats.json",
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
)

print(f"Proceso completo sin errores. RMS global = {rms:.4f}")
import sys; sys.exit()


# pruebas
ds = PackedPile()
loader = DataLoader(ds, batch_size=2)

for i, batch in enumerate(loader):
    print(batch["input_ids"].shape)
    if i == 2:
        break
# find e pruebas