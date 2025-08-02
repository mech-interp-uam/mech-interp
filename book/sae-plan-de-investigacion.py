#!/usr/bin/env python
# coding: utf-8

# # Entrenamiento de un SAE sobre las salidad y activaciones de la MLP intermedia de llama3.2 1B

# ## Costo de entrenamiento
# 
# Generalmente, el costo computacional de un LLM está dominado por la evaluación
# de sus MLPs (referencia a el seminario en una universidad de un ex empleado
# de anthropic)
# 
# Una aprocimación simple del costo de entrenamiento de un modelo es
# 
# $$
#   6ND 
# $$
# esto en términos de FLOPs (operaciones de punto flotante).
# (citar a chinchilla scaling laws)
# 
# donde $N$ es el número de parámetros y $D$ la cantidad de muestras en el
# conjunto de entrenamiento.
# 
# Esto se debe a que, generalmente, cada parámetro actua en una multiplicaciónl
# y en una suma de punto flotante, dandonos un costo de $2ND$ tan solo en el
# forward pass. Tipicamente, el costo de el backward pass es el doble del forward
# pass, haciendo que su costo sea $4ND$. Sumando tenemos el resultado previamente
# mencionado.
# 
# Sea $N_l$ el número de parámetros de llama.
# Como nosotros vamos a entrenar un autoencoder sobre un MLP a la mitad de llama,
# solo necesitamos evaluar esa primera mitad. Además, no correremos el backward
# pass sobre los parámetros de llama, pues no buscamos modificarlos, es decir, los
# mantendrémos fijos. Por esto, tenemos que el FLOPs realizados por tal mitad del
# modelo llama es
# 
# $$
#   N_l D
# $$
# 
# En cuanto al SAE, solo considerando el costo de aplicar sus matrices, tenemos
# 
# $$
#   6 (2d_\text{in}d_\text{sae})D
# $$

# En el caso de gemmascope, entre todos los SAEs que entrenaron, los más pequeños
# entrenados en las salidas de las MLPs, se entrenaron en 4 mil millones de
# vectores de activaciones, con la dimencion de los vectores en el stream
# recidual (y por lo tanto, de la salida de las capas MLP), es 2048, con
# $d_\text{sae} = 2028 * 8$.
# 
# Deseamos encontrar hiperparámetros para entrenar SAEs, para esto:
# - Usamos una primera aproximación razonable modificando los hiperparámetros para
#   los autoencoders más pequeños entrenados en salidas de las MLPs de gemma 2
#   2 B
# - Ajustamos una power law en base a 2 entrenamientos de SAEs más pequeñas,
#   usando el mismo learning rate.
# - Ajustamos una power law para el learning rate con los hiperparámetos optimos
#   que estimó el paso anterior.
# 
# Si ignoramos la relación posicional de los tokens y asumimos una distribución
# uniforme, tenemos que la entropía por token es
# 
# $$
#   \log_2 (\text{tamaño del vocabulario})
# $$
# ya que el vocabulario de gemma2 2B es 256000 y el de llama es 128000, obtenemos
# que el número de tokens equivalente sería 4.2 mil millones. Ya que nuestro
# modelo es la mitad de tamaño de gemma2 2B, una primera cantidad de datos
# razonable para entrenar nuesto sae más grande es $2.1B$
# 
# En el caso de llama3.2 1B, eso resultaría en
# 
# $$
#   N_l D = (2.1 \times 10^9)(1.2 \times 10^9) \approx 2.5 \times 10^{18}
# $$
# Una RTX 4090 puede realisar cada segundo un máximo de $165 \times 10^{12}$
# operaciones con tensores de 16 bits y acumulador de 32 bits (referencia
# al reporte técnico v1.0.1), luego, estimamos 4.2 horas de entrenamiento
# tan solo considerando la computación en el modelo llama.
# 
# Ahora, para estimar las horas-RTX4090 para el autoencoder, en el caso de
# entrenarlo en la salida de la MLP intermedia, tendríamos
# 
# $$
#     6(2.1 \times 10^9)(2048^2)(8)(2) = 8.5 \times 10^{17}
# $$

# # El modelo

# In[ ]:


import torch
from torch import nn
from jaxtyping import Array, Float, Int
import dataclasses
from tqdm import tqdm
from torch.nn import functional as F
from torch.linalg import vector_norm
import math
from datasets import load_dataset
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.profiler as prof 
from contextlib import nullcontext
import matplotlib.pyplot as plt
from math import sqrt
import argparse


# In[ ]:


dtype = torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--use-d-model-std', action='store_true', 
                    help='Use 1/sqrt(d_model) for weight std instead of 1/sqrt(d_sae)')
parser.add_argument('--run-name', type=str, default="b4kwarmup2kdataworkers12bf16exp24",
                    help='Name for the training run (used for logging directories)')
parser.add_argument('--exp-factor', type=int, default=24,
                    help='Expansion factor for d_sae = d_model * exp_factor')
parser.add_argument('--sparsity-coeff', type=float, default=0.001,
                    help='Maximum sparsity coefficient')
parser.add_argument('--total-steps', type=int, default=256000,
                    help='Total number of training steps')
parser.add_argument('--sparsity-warmup-full', action='store_true',
                    help='Use full training length for sparsity warmup instead of 100k steps')
parser.add_argument('--max-lr', type=float, default=14e-5,
                    help='Maximum learning rate')
parser.add_argument('--seed', type=int, default=42,
                    help='Random seed for reproducibility')
parser.add_argument('--compile-mode', type=str, default='max-autotune', 
                    choices=['max-autotune', 'default', 'none'],
                    help='PyTorch compile mode: max-autotune, default, or none')
parser.add_argument('--early-exit', type=int, default=0,
                    help='Exit training after N steps (0 = no early exit)')
parser.add_argument('--log-timing', action='store_true',
                    help='Log epoch timing to file (logged after each epoch to avoid GPU sync)')
parser.add_argument('--deterministic', action='store_true',
                    help='Enable deterministic behavior (may affect performance)')
args = parser.parse_args()

# Set random seed for reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# Set deterministic behavior if requested
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

USE_PROFILER = True
d_model = 2048
d_sae = d_model * args.exp_factor
w_std = 1/sqrt(d_model) if args.use_d_model_std else 1/sqrt(d_sae)


# In[ ]:


class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        bandwidth = 0.001
        x, threshold = ctx.saved_tensors
        mask = (abs(x - threshold) < bandwidth/2) & (x > 0)
        grad_threshold = -1.0/bandwidth * mask.to(x.dtype)
        return torch.zeros_like(x), grad_threshold * grad_output


# In[ ]:


class Sae(nn.Module):
    def __init__(self, d_model, d_sae, use_pre_enc_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.enc = nn.Linear(d_model, d_sae, dtype=dtype)
        self.dec = nn.Linear(d_sae, d_model, dtype=dtype)
        w = torch.randn(d_model, d_sae)
        w *= ((w_std * sqrt(d_model))/vector_norm(w, dim=0, keepdim=True))
        with torch.no_grad():
            # normalize each of the d_sae dictonary vectors
            self.dec.weight.copy_(w.clone().to(dtype))
            self.enc.weight.copy_(w.clone().to(dtype).t())
            self.enc.bias.copy_(torch.zeros_like(self.enc.bias))
            self.dec.bias.copy_(torch.zeros_like(self.dec.bias))
        self.log_threshold = nn.Parameter(
            torch.log(torch.full((d_sae,), 0.001, dtype=dtype)))
        self.use_pre_enc_bias = use_pre_enc_bias
        def project_out_parallel_grad(dim, tensor):
            @torch.no_grad
            def hook(grad_in):
                # norm along dim=dim of the tensor is assumed to be w_std * sqrt(d_model) as we
                # are going to normalize it after every grad update
                norm_factor = w_std * sqrt(d_model)
                dot = (tensor * grad_in).sum(dim=dim, keepdim=True)
                return grad_in - dot * tensor / (norm_factor * norm_factor)
            return hook

        self.dec.weight.register_hook(
            project_out_parallel_grad(0, self.dec.weight))
                

    def forward(self,
        x,
    ):
        "We compute this much here so that compile() can do its magic"
        # as per train_gpt2.py on karpathy's llm.c repo, there are performance
        # reasons not to return stuff
        d = {}
        original_input = x
        if self.use_pre_enc_bias:
            x = x - self.dec.bias
        
        x = self.enc(x)
        threshold = torch.exp(self.log_threshold)
        s = Step.apply(x, threshold)
        x = x*s
        x = self.dec(x)
        d['mask'] = s
        d['reconstruction'] = ((x - original_input).pow(2)).mean(0).sum()

        return d


# In[ ]:


def cosine_schedule_with_warmup(
    current_step: int,
    warmup_steps: int,
    total_steps: int
    ):
    if current_step < warmup_steps:
        # Linear warmup from 10% to 100%
        lr = 0.1 + 0.9 * current_step / warmup_steps
        return lr
    progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
    lr =  0.5 * (1 + math.cos(math.pi * progress))
    return lr


# In[ ]:


def sparsity_schedule(step, warmup_steps, max_sparsity_coeff):
    if step >= warmup_steps:
        return max_sparsity_coeff
    return max_sparsity_coeff*((step+1) / warmup_steps)


# In[ ]:



# Load the dataset with all train_* splits
from datasets import concatenate_datasets

# Load all train_* splits
dataset_dict = load_dataset('naraca/activaciones-llama3-mlp8')
train_splits = [split for split in dataset_dict.keys() if split.startswith('train_')]
print(f"Found {len(train_splits)} train splits: {train_splits[:5]}...")

# Concatenate all train splits
train_datasets = [dataset_dict[split] for split in train_splits]
ds_train = concatenate_datasets(train_datasets)
ds = {'train': ds_train}
ds['train'].set_format('numpy')

# Check the dataset structure
print(f"Dataset loaded successfully: {len(ds['train'])} examples")
print(f"Features: {ds['train'].features}")
print(f"First example length: {len(ds['train'][0]['activacion'])}")


# In[ ]:


#B = 1024 * 4
n = 100
# sample_norms = (
#     ds['train']
#     .batch(B)
#     .shuffle()
#     .take(n)
#     .map(lambda row: {"norm": np.linalg.norm(row["activations"], axis=1).mean()},
#          remove_columns=['activations'])
# )
# norm = None
# for i, sample_norm in enumerate(sample_norms):
#     current_norm = sample_norm['norm']
#     if i == 0:
#         norm = current_norm
#         continue
#     norm = i/(i+1) * norm + 1/(i+1) * current_norm
# 
# print(norm)


# In[ ]:



class ActivationsDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __getitem__(self, idx):
        # Return as float16, on modern GPUs conversion from float 16 to 32 is
        # free compared to matmults or so I was told
        # Convert from list to numpy array, then from uint16 to bfloat16
        activations = self.data[idx]['activacion']
        activations_np = np.array(activations, dtype=np.uint16)
        activations_bf16 = torch.from_numpy(activations_np).view(torch.bfloat16)
        return activations_bf16.float()

    def __len__(self):
        return len(self.data)


# In[ ]:


batch = 1024 * 4
dataset = ActivationsDataset(ds['train'])
dataloader = DataLoader(
    dataset,
    batch_size=batch,
    shuffle=True,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=128,
    persistent_workers=False,
    drop_last=True,
)


# In[ ]:


len(dataloader)


# In[ ]:


torch.set_float32_matmul_precision('high')
steps = 2**16
max_lr = args.max_lr
model = Sae(d_model, d_sae)
model.to(device)
# Apply compile mode based on CLI argument
if args.compile_mode != 'none':
    model.compile(
            mode=args.compile_mode,
            dynamic=False,
            fullgraph=True,
    #         options = {
    #             "max_autotune":True,
    #             "epilogue_fusion":True,
    #             "triton.cudagraphs":True,
    #             },
            )
warmup_steps=2000
sparcity_warmup_steps = args.total_steps if args.sparsity_warmup_full else 100000
total_steps=args.total_steps
optimizer = torch.optim.Adam([
    {"params": model.enc.parameters(), "lr":   max_lr, "betas":(0.0, 0.999)},
    {"params": model.dec.parameters(), "lr":   max_lr, "betas":(0.0, 0.999)},
    {"params": model.log_threshold,    "lr": 1*max_lr, "betas":(0.0, 0.999)},
], fused=True)
max_sparsity_coeff = args.sparsity_coeff

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_schedule_with_warmup(step, warmup_steps, total_steps)
)

run_name = args.run_name
writer = SummaryWriter(f"runs/{run_name}")

# Create directory for external logging
import os
os.makedirs(f"runs/{run_name}/prevalences", exist_ok=True)
os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)

# Initialize timing variables
import time
epoch_start_time = None
epoch_number = 0


should_break = False
total_step = 0

steps_per_tensorboard_log = 1
prevalences_moving_average = 0.0
# we need to detect prevalences of at 1 every 10_000_000
prevalences_moving_average_batches = 2000
prevalences_moving_average_averaged = 0
eval_prevalences_every = 5000

bin_edges = np.logspace(np.log10(1e-8), np.log10(100), num=101).tolist()

# Save bin edges once at start
np.save(f"runs/{run_name}/prevalences/bin_edges.npy", np.array(bin_edges))

plot_features_every = 1000

def heatmap_feature_products(features):
    ...

def feature_dimentionalities(features: Float[Array, "d_model d_sae"]) -> Float[Array, "d_sae"]:
    # careful, this takes a lot of vram,
    dot_products = features.T @ features
    return dot_products.diag()/dot_products.norm(dim=0)

cmap = plt.get_cmap("viridis")



profiler = prof.profile(
    schedule=prof.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=prof.tensorboard_trace_handler("runs/prof"),
    record_shapes=True,
    with_flops=True,
    with_stack=True,
    ) if USE_PROFILER else nullcontext()

training_ctx = profiler if USE_PROFILER else nullcontext()
with training_ctx:
    while total_step < total_steps:
        epoch_start_time = time.time()
        for step, x in enumerate(tqdm(dataloader)):
            x = x.to(device, non_blocking=True).to(dtype)
            x /= 3.4 # this is supposed to be the expected norm
            optimizer.zero_grad()
            # you can do without the prevalences, use a rolling average for
            # mask that you clean once in a while and sent or do stuff when it
            # has averaged over enough steps
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                d = model(
                        x,
                        #return_prevalences=True,
                        # # for accuracy, use prevalences to compute the sum of s outside autocast
                        # return_l0=False,
                        )
            reconstruction_loss, mask = d['reconstruction'], d['mask']
            prevalences = mask.mean(0)
            l0 = prevalences.sum()
            # l0 = active_latent_ratio * d_sae

            # compute the prevalence of each neuron
            with torch.no_grad():
                if total_step % eval_prevalences_every == 0:
                    # use next prevalences_moving_average_batches steps to eval the
                    # prevalences

                    # init
                    evaluating_prevalences = True
                    prevalences_moving_average = 0.0
                    prevalences_moving_average_averaged = 0

                if evaluating_prevalences:
                    prevalences_moving_average = (
                        (1./(prevalences_moving_average_averaged+1)) * prevalences
                        +
                        (1. - 1/(prevalences_moving_average_averaged+1)) * prevalences_moving_average
                    )
                    prevalences_moving_average_averaged += 1


                    if prevalences_moving_average_averaged >= prevalences_moving_average_batches:
                        evaluating_prevalences = False
                        writer.add_histogram("prevalences", 100. * prevalences_moving_average, total_step,
                                bins=bin_edges
                                )
                        writer.add_histogram("log10 prevalences", torch.log10(prevalences_moving_average + 1e-8), total_step)
                        
                        # Save raw prevalence data asynchronously
                        prevalences_copy = prevalences_moving_average.detach().float().clone()
                        import threading
                        def save_prevalences():
                            prevalences_cpu = prevalences_copy.cpu().numpy()
                            np.save(f"runs/{run_name}/prevalences/step_{total_step}.npy", prevalences_cpu)
                        threading.Thread(target=save_prevalences, daemon=True).start()
                        # print(total_step)

                        percent_dead = 100. * (prevalences_moving_average < 1e-7).to(dtype).mean()
                        # Should we need to call .detach().cpu().item()?
                        writer.add_scalar("percent dead", percent_dead, total_step)

                if total_step % plot_features_every == 0:
                    to_plot = 256
                    features = model.dec.weight
                    dot_products = features.T @ features
                    dimentionalities = dot_products.diag()/dot_products.norm(dim=0)
                    dot_products = dot_products.fill_diagonal_(0)
                    idx = torch.argsort(dimentionalities, descending=True)[:256]
                    dot_products = dot_products[idx]
                    dot_products = dot_products[:, idx]
                    dot_products = dot_products / dot_products.max()
                    dot_products = cmap(dot_products.detach().cpu().numpy())[...,:3]
                    dot_products = torch.as_tensor(dot_products).permute(2 ,0 ,1)
                    writer.add_image("dot products", dot_products, total_step)




            

            sparsity_coefficient = sparsity_schedule(total_step, sparcity_warmup_steps, max_sparsity_coeff)
            loss = reconstruction_loss + sparsity_coefficient * l0
            # log losses, compute stats, etc
            grad = loss.backward()
            # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # metrics
            if (total_step % 5000) == 0:
                with torch.no_grad():
                    # print metrics
                    print(f"{total_step=}")
                    print(f"reconstruction={reconstruction_loss.item()}")
                    print(f"l0={l0.item()}")
                    # print(f"norm={norm.item()}")
                    print(f"{sparsity_coefficient=}")
            if (total_step % steps_per_tensorboard_log) == 0:
                # Use .detach() to avoid GPU-CPU sync during training
                writer.add_scalar(
                        "Reconstruction loss/train",
                        reconstruction_loss.detach(),
                        total_step,)
                writer.add_scalar("L0 loss/train",
                        l0.detach(),
                        total_step)
                writer.add_scalar("lr",
                        scheduler.get_last_lr()[0],
                        total_step)
                writer.add_scalar("sparsity coefficient",
                        sparsity_coefficient,
                        total_step)
            optimizer.step()
            # TODO: sparsity_coefficient scheduler
            # print(scheduler.get_last_lr()[0])
            scheduler.step()

            # normalize
            with torch.no_grad():
                model.dec.weight *= ((w_std * sqrt(d_model))/vector_norm(model.dec.weight, dim=0, keepdim=True))
        # print(f"epoch loss: {loss.detach().item()}")
            
            # Profiler step
            if USE_PROFILER:
                profiler.step()
            
            # Save model checkpoint every 5000 steps after 40k steps (asynchronously)
            if total_step % 5000 == 0 and total_step >= 40000:
                import threading
                def save_checkpoint():
                    checkpoint_path = f"runs/{run_name}/checkpoints/step_{total_step}.pth"
                    torch.save(model.state_dict(), checkpoint_path)
                threading.Thread(target=save_checkpoint, daemon=True).start()
                
            total_step +=1
            if total_step > total_steps:
                should_break = True
                break
            
            # Early exit if requested
            if args.early_exit > 0 and total_step >= args.early_exit:
                print(f"Early exit at step {total_step}")
                should_break = True
                break
        # Log epoch timing if requested
        if args.log_timing:
            epoch_duration = time.time() - epoch_start_time
            with open(f"runs/{run_name}/epoch_timing.txt", "a") as f:
                f.write(f"Epoch {epoch_number}: {epoch_duration:.3f}s\n")
            epoch_number += 1
            
        if should_break:
            break



# In[ ]:


# Save model with run name to avoid overwriting
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), f"models/sae_{run_name}.pth")


# In[ ]:


sae2 = Sae(d_model, d_sae)
sae2.load_state_dict(model.state_dict())


# In[ ]:



