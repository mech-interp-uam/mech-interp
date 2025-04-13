# extraer_mlp8_hf.py
from huggingface_hub import login, HfApi
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import os
import json
import zipfile
import shutil

# === CONFIGURACIÓN ===
login("TU_TOKEN_AQUI")

repo_id = "naraca/mi-dataset-activaciones-llama3_2"
model_name = "meta-llama/Llama-3.2-1B"
jsonl_path = "textos_oscar_1porciento.jsonl"
batch_size = 32
max_length = 256
layer_idx = 8
zip_every = 1000

# === PREPARACIÓN ===
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    token="TU_TOKEN_AQUI"
)
model.eval()

mlp_outputs = []

def capture_mlp_output(module, input, output):
    mlp_outputs.append(output.detach().cpu())

model.model.layers[layer_idx].mlp.register_forward_hook(capture_mlp_output)

# === CARGAR TEXTOS ===
texts = []
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        texts.append(json.loads(line)["text"])

from torch.utils.data import DataLoader

def collate_fn(batch_texts):
    return tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)

loader = DataLoader(texts, batch_size=batch_size, collate_fn=collate_fn)

# === FUNCIONES DE SUBIDA ===
api = HfApi()
temp_folder = "activaciones_mlp8_temp"
os.makedirs(temp_folder, exist_ok=True)

def subir_zip_a_huggingface(path_zip):
    print(f"⬆ Subiendo {path_zip} a Hugging Face...")
    api.upload_file(
        path_or_fileobj=path_zip,
        path_in_repo=os.path.basename(path_zip),
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f" Subido: {path_zip}")

# === LOOP PRINCIPAL ===
with torch.no_grad():
    lote_idx = 0
    for i, batch in enumerate(loader):
        try:
            batch = {k: v.to(model.device) for k, v in batch.items()}
            mlp_outputs.clear()
            _ = model(**batch)

            for j, tensor in enumerate(mlp_outputs):
                idx_global = i * batch_size + j
                fname = os.path.join(temp_folder, f"lote_{idx_global:06d}.npy")
                np.save(fname, tensor.numpy())

            if (i + 1) % 10 == 0:
                print(f" Procesados y guardados {((i + 1) * batch_size)} textos")

            # Subir cada zip_every ejemplos
            if (i + 1) * batch_size % zip_every == 0:
                zip_name = f"mlp8_lote_{lote_idx:03d}.zip"
                shutil.make_archive(zip_name.replace(".zip", ""), 'zip', temp_folder)
                subir_zip_a_huggingface(zip_name)
                shutil.rmtree(temp_folder)
                os.makedirs(temp_folder, exist_ok=True)
                os.remove(zip_name)
                lote_idx += 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f" GPU OOM en el lote {i}, limpiando caché...")
                torch.cuda.empty_cache()
            else:
                raise e

# Subir lo restante
if len(os.listdir(temp_folder)) > 0:
    zip_name = f"mlp8_lote_{lote_idx:03d}.zip"
    shutil.make_archive(zip_name.replace(".zip", ""), 'zip', temp_folder)
    subir_zip_a_huggingface(zip_name)
    shutil.rmtree(temp_folder)
    os.remove(zip_name)

print(" Proceso completo sin errores.")
#
