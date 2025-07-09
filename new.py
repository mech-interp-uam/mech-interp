#!/usr/bin/env python
# extraer_mlp8_streaming.py  – listo para entrenar SAE (GemmaScope + RMS)

from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset, Features, Sequence, Value
from torch.utils.data import IterableDataset, DataLoader
import torch, itertools, gc, os, json, tempfile, math

HF_TOKEN   =        
REPO_ID    = "mech-interp-uam/llama3.2-1b-sae"
MODEL_NAME = "meta-llama/Llama-3.2-1B"
SAMPLE_DOCS = 5_000 # cambiar a 100_000
MAX_LEN     = 4096
BATCH_SIZE  = 2
LAYER_IDX   = 8
SHARD_SIZE  = 1_000

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
    mlp_out = out.detach().cpu()
model.model.layers[LAYER_IDX].mlp.register_forward_hook(hook_mlp) #metemos el hohok a la capa 

# valid_sources = {"Wikipedia (en)", "StackExchange", "PubMed Abstracts", "ArXiv", }
                 
raw_ds = load_dataset(
    "monology/pile-uncopyrighted",
    split="train",
    streaming=True,
    download_mode="reuse_dataset_if_exists",  # o force_redownload si prefieres
    trust_remote_code=True,
).filter(lambda x: (
    (txt := x["text"].strip()) != "" and
    len(txt) > 20 and
    sum(c.isalpha() for c in txt) >= 2 and
    txt.upper() != txt and
    not txt.isnumeric()
)).shuffle(buffer_size=50_000, seed=42)

eos = tokenizer.eos_token_id
bos = tokenizer.bos_token_id

# raw_ds = raw_ds.batch(8190).map(lambda x: [[[eos] + y + [bos] for y in tokenizer(x["text"], add_special_tokens=False)]])
# raw_ds = itertools.islice(raw_ds, SAMPLE_DOCS)

def tokenize_batch(batch):
    batch =  tokenizer(
        batch["text"],
        add_special_tokens=False).input_ids
    print(type(batch))
    l = []
    for seq in batch:
        l.append([bos] + seq + [eos])
    return {"text": l}
raw_ds = raw_ds.map(tokenize_batch, batch_size=1000, batched=True, remove_columns=["text"])
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
    iterador = iter(raw_ds)                 # convertimos d.s a iterador
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
        buf = []                               # donde se van pegando tokens
        for tid in token_stream2(raw_ds):             #     token_stream emite UN token a la vez
            buf.append(tid)                    # añade el token al final del buffer
            while len(buf) >= MAX_LEN:         # ¿ya cabe un bloque entero?
                yield self._make_item(buf[:MAX_LEN])   # saca los PRIMEROS max_len
                buf = buf[MAX_LEN:]            # deja en buf lo que sobró
       
    def _make_item(self, ids):
        ids = torch.tensor(ids, dtype=torch.long) #tokens a tensor
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}         # deja en buf lo que sobró


loader = DataLoader(PackedPile(), batch_size=BATCH_SIZE, num_workers=0)
# model() espera un tensor de ENTEROS o dict de tensores de enteros 
features = Features({ #estructura de cada ejemplo, su valor es una secuencia de vectores uno por token
    "activacion": Sequence(feature=Sequence(Value("float16"))) #tamaño hidden_dim
}) # 

import ramdon
buffer, shard_idx, n_proc = [], 0, 0
sum_sq, n_tokens = 0.0, 0          # para RMS

with torch.no_grad(): 
    for batch in loader: # toma batch del DL cada bthc dic inputo_ids y attention_mask (b_s, m_l)
        batch = {k: v.to(model.device) for k, v in batch.items()} #mueve los tens
        mlp_out = None
        inp, attn = batch["input_ids"], batch["attention_mask"]
        _ = model(input_ids=inp, attention_mask=attn)
        # aqyu batch tiene que ser matrix cuadrada de enteros
        #_ = model(batch)

        for i in range(batch["input_ids"].shape[0]):
            ids  = batch["input_ids"][i].cpu()
            # 1. Detectar las pociciones de la entrada donde hay token especial
            # vamos a guardar a disco las activaciones cuya pocision no sea la misma
            # que algun token especial en la entrada
            # por ejemplo
            # crear una mascara de shape igual a inputs_ids.flaten donde sea 1 si es un token
            # especial y 0 si no
            # aplanar la salida del modelo, indexar filtrando usando la mascara
            # guardar a disco
            acts = mlp_out[i]                      # (seq, hid)
            keep = ~torch.isin(ids, torch.tensor(tokenizer.all_special_ids))
            # isin(ids flatten)
            sel  = acts[keep].to(torch.float16)    # (m, hid)
            

            # actualización RMS (usa fp32 para precisión)
            sum_sq += sel.to(torch.float32).pow(2).sum(dim=1).item()
            # ir promediando
            # total_promediadas = 10
            # running_norm_sqr = 9/10 * running_norm_sqr + 1/10 * norm_actual_sqr
            # totam_promediado += 1
            n_tokens += sel.shape[0]

            buffer.append({"activacion": sel.half().cpu().numpy()})

            n_proc += 1

        if len(buffer) >= SHARD_SIZE:
            random.shuffle(buffer)
            Dataset.from_list(buffer, features=features)\
                   .push_to_hub(REPO_ID, split="train", private=True,
                                max_shard_size="500MB")
            print(f"Shard {shard_idx} subido — {n_proc} ejemplos")
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