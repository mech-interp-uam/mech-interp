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
#
# the configuration foound to work with fp16 training the
# best is
# x ~ N(0, 64/d_model)
# with d_sae either 2048*32 or 2048*24
# W ~ N(0, 64/d_sae) that is N(0, 1/1024) (the matrix the encoder and
# decoder have in common at initialization time)
# we suspect this avoids underfows
# what follows is a carefull analysis if the forwards and backwards graph, in an
# attempt to align theory the practice
#
# The product of such entry of x and of W is has variance
#
# Var(x) Var(w) = 64/d_model * 64/d_sae aprox 0.01
#
# Then, we add d_model of those together, to get an that the preactivations have
# variance:
#
# Var(x) d_model * 64/d_model 64/d_sae = 64**2/d_sae aprox 1/4
#
# then, if the l2^2 is between 0.4 and 0.02, and is a sum of
# d_model elemets per thing in the batch, then each is about
# 0.01 to 0.001 on average per element, thus it represents
# no underflow risk

# now let's think of the backwards pass:
# the l0 loss: we warm it up from 0 to l0*0.0009 during training,
# thus, on the decoder branch, we get all the full loss, which starts at
# about d_sae/2, grows for some steps, then comes down rapidly
# thus, first it gets spread over all d_model elemens, that is
# d_sae/(2 d_model), thus about 12 or 16
# then it gets mult by a special scalar designed to scale the gradient
# contributions properly to stabilize training: 1/x_mult**2 so 1/64
# then it gets scaled up by 2 (because of squaring)
# so at this point each float is around 6/64 or 8/64
# which is 0.9 to 0.12
# so if we think of that as the std
# then the variance is 0.009 to 0.01
# finally, it passes the backward of whose forwards is
# acts -> acts W^t, that is
# grad_out -> grad_out @ self.dec.weight
# at this point its variance is
#
# (0.009 to 0.012) * 64 / d_sae 
#
# thus, here we have an std of 0.006 to 0.007
# confortably represented as a normal float16
# once you have those multiplications, don't forget to sum!
# each latent neuron gets contributions from d_model things, so we multiply our
# variance above
# current var: from 12  to 14 let's round to 13
# then, we pass this grad throud the jumprelu, which
# just passes the gradient or zeroes depending on the threshold set by the Step
# function, at the begining of thraining roughly half of the gradients are
# blocked thus after this we have a variance of of 7.5 still safe.
# we follow by multiplying with the encoder matrix whose entries have variance
# 64/d_sae, thus after this multiplication of floats we have
# current var: 7.5 * 64/ d_sae
# or a std of 0.2
# then each float in the input to the encoder gets contributions from all the
# ACTIVE latent neurons, do d_sae / 2
# we sum all of those identical thinkgs to ger a variance
# d_sae/2 * 7.5 * 64/d_sae = 7.5 * 32
# or a std of 15.5, which is large but does not cause overflow in float16.
# finally, we scale (up this time) by that special scalar, giving us a (without
# squaring this time) (which is 8)
# almost cancelling the 7.5 and resulting on a variance of 32
# thus, we did avoid underflows and overflows.


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


# In[ ]:


dtype = torch.float16
lower_dtype = torch.float16
device = "cuda" if torch.cuda.is_available() else "cpu"
USE_PROFILER = False


# In[ ]:


norm_mult = 8.
x_mult = 8.
class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        bandwidth = 0.001*norm_mult*x_mult
        x, threshold = ctx.saved_tensors
        mask = (abs(x - threshold) < bandwidth/2) & (x > 0)
        grad_threshold = -1.0/bandwidth * mask.to(x.dtype)
        return torch.zeros_like(x), grad_threshold * grad_output


# In[ ]:


class Sae(nn.Module):
    def __init__(self, d_in, d_sae, use_pre_enc_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.enc = nn.Linear(d_in, d_sae, dtype=dtype)
        self.dec = nn.Linear(d_sae, d_in, dtype=lower_dtype)
        with torch.no_grad():
            # normalize each of the d_sae dictonary vectors
            self.dec.weight /= vector_norm(self.dec.weight, dim=1, keepdim=True)
            self.dec.weight *= norm_mult
            self.enc.weight.copy_(self.dec.weight.clone().t().to(self.enc.weight.dtype))
            self.enc.bias.copy_(torch.zeros_like(self.enc.bias))
            self.dec.bias.copy_(torch.zeros_like(self.dec.bias))
        self.log_threshold = nn.Parameter(
            torch.log(torch.full((d_sae,), 0.001 * norm_mult*x_mult, dtype=dtype)))
        self.use_pre_enc_bias = use_pre_enc_bias
        def project_out_parallel_grad(dim, tensor):
            @torch.no_grad
            def hook(grad_in):
                # norm along dim=dim of the tensor is assumed to be 1 as we
                # are going to normalize it after every grad update
                dot = (tensor * grad_in).sum(dim=dim, keepdim=True)
                return grad_in - dot * tensor
            return hook

        self.dec.weight.register_hook(
            project_out_parallel_grad(1, self.dec.weight))
                

    def forward(self,
        x,
    ):
        "We compute this much here so that compile() can do its magic"
        # as per train_gpt2.py on karpathy's llm.c repo, there are performance
        # reasons not to return stuff
        d = {}
        original_input = x
        if self.use_pre_enc_bias:
            x = x - self.dec.bias.to(x.dtype)
        
        x = self.enc(x)
        threshold = torch.exp(self.log_threshold)
        s = Step.apply(x, threshold)
        x = x*s
        # print((x==0).sum())
        # x *= 100.
        x = x.to(lower_dtype)
        # import sys;sys.exit() 
        # x = self.dec(x)
        x = x @ self.dec.weight.t()
        x += self.dec.bias
        x = x.to(original_input.dtype)
        # x /= 100.
        x /= norm_mult**2
        d['mask'] = s
        d['reconstruction'] = ((x - original_input).pow(2)).mean(0).sum()/x_mult**2

        return d


# In[ ]:


def cosine_schedule_with_warmup(
    current_step: int,
    warmup_steps: int,
    total_steps: int
    ):
    if current_step < warmup_steps:
        lr =  (1 + current_step) / warmup_steps
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



# Load the dataset
ds = load_dataset('mech-interp-uam/llama-mlp8-outputs')
ds.set_format('numpy')

# Check the dataset structure
print(f"Dataset loaded successfully: {len(ds['train'])} examples")
print(f"Features: {ds['train'].features}")
print(f"First example shape: {ds['train'][0]['activations'].shape}")


# In[ ]:


B = 1024
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
        return torch.tensor(self.data[idx]['activations'])

    def __len__(self):
        return len(self.data)


# In[ ]:


batch = 1024
dataset = ActivationsDataset(ds['train'])
dataloader = DataLoader(
    dataset,
    batch_size=batch,
    shuffle=True,
    num_workers=12,
    pin_memory=True,
    prefetch_factor=128,
    persistent_workers=True,
    drop_last=True,
)


# In[ ]:


len(dataloader)


# In[ ]:


torch.set_float32_matmul_precision('high')
steps = 2**16
max_lr = 7e-5
d_in = 2048
d_sae = 2048*32
model = Sae(d_in, d_sae)
model.to(device)
model.compile(
        mode="max-autotune",
        dynamic=False,
        fullgraph=True,
#         options = {
#             "max_autotune":True,
#             "epilogue_fusion":True,
#             "triton.cudagraphs":True,
#             },
         )
warmup_steps=1000
sparcity_warmup_steps = 256000
total_steps=256000 #for now
optimizer = torch.optim.Adam([
    {"params": model.enc.parameters(), "lr":   max_lr, "betas":(0.0, 0.999)},
    {"params": model.dec.parameters(), "lr":   max_lr, "betas":(0.0, 0.999)},
    {"params": model.log_threshold,    "lr": 1*max_lr, "betas":(0.0, 0.999)},
], fused=True)
max_sparsity_coeff = 0.0009

scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_schedule_with_warmup(step, warmup_steps, total_steps)
)

writer = SummaryWriter("/workspace/runs/allfp16x_mult8scale8warmup1000beta999v3x32compile")

should_break = False
total_step = 0

steps_per_tensorboard_log = 1
prevalences_moving_average = 0.0
# we need to detect prevalences of at 1 every 10_000_000
prevalences_moving_average_batches = 20000
prevalences_moving_average_averaged = 0
eval_prevalences_every = 500000

bin_edges = np.logspace(np.log10(1e-8), np.log10(100), num=101).tolist()

plot_features_every = 0
plot_params_every = 5000

def heatmap_feature_products(features):
    ...

def feature_dimentionalities(features: Float[Array, "d_model d_sae"]) -> Float[Array, "d_sae"]:
    # careful, this takes a lot of vram,
    dot_products = features.T @ features
    return dot_products.diag()/dot_products.norm(dim=0)

cmap = plt.get_cmap("viridis")



training_ctx = nullcontext() if not USE_PROFILER else prof.profile(
    schedule=prof.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=prof.tensorboard_trace_handler("runs/prof"),
    record_shapes=True,
    with_flops=True,
    with_stack=True,
    )
with training_ctx:
    while total_step < total_steps:
        for step, x in enumerate(tqdm(dataloader)):
            x = x.to(device, non_blocking=True).to(dtype)
            x *= x_mult
            x /= 3.4 # this is supposed to be the expected norm
            optimizer.zero_grad()
            # you can do without the prevalences, use a rolling average for
            # mask that you clean once in a while and sent or do stuff when it
            # has averaged over enough steps
            #with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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
                        # print(total_step)

                        percent_dead = 100. * (prevalences_moving_average < 1e-7).to(dtype).mean()
                        # Should we need to call .detach().cpu().item()?
                        writer.add_scalar("percent dead", percent_dead, total_step)

                if (plot_features_every > 0) and (total_step % plot_features_every) == 0:
                    to_plot = 256
                    features = model.dec.weight
                    dot_products = features.T @ features
                    dimentionalities = dot_products.diag()/dot_products.norm(dim=0)
                    dot_products = dot_products.fill_diagonal_(0)
                    idx = torch.argsort(dimentionalities, descending=True)[:256]
                    dot_products = dot_products[idx]
                    dot_products = dot_products[:, idx]
                    dot_products = dot_products / dot_products.max()
                    dot_products = cmap(dot_products.detach().float().cpu().numpy())[...,:3]
                    dot_products = torch.as_tensor(dot_products).permute(2 ,0 ,1)
                    writer.add_image("dot products", dot_products, total_step)

                if total_step % plot_params_every == 0:
                    writer.add_histogram("wdec", model.dec.weight, total_step)
                    writer.add_histogram("wenc", model.enc.weight, total_step)




            

            sparsity_coefficient = sparsity_schedule(total_step, sparcity_warmup_steps, max_sparsity_coeff)
            loss = reconstruction_loss + sparsity_coefficient * l0
            # log losses, compute stats, etc
            grad = loss.backward()
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
                writer.add_scalar(
                        "Reconstruction loss/train",
                        reconstruction_loss.item(),
                        total_step,)
                writer.add_scalar("L0 loss/train",
                        l0.item(),
                        total_step)
                writer.add_scalar("lr",
                        scheduler.get_last_lr()[0],
                        total_step)
                writer.add_scalar("sparsity coefficient",
                        sparsity_coefficient,
                        total_step)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), .9)
            writer.add_scalar("grad_norm", grad_norm, total_step)
            optimizer.step()
            # TODO: sparsity_coefficient scheduler
            # print(scheduler.get_last_lr()[0])
            scheduler.step()

            # normalize
            with torch.no_grad():
                wdecnorm = vector_norm(model.dec.weight, dim=1, keepdim=True, dtype=dtype).to(lower_dtype)
                model.dec.weight /= wdecnorm
                model.dec.weight *= norm_mult
        # print(f"epoch loss: {loss.detach().item()}")
            total_step +=1
            if total_step > total_steps:
                should_break = True
                break
        if should_break:
            break



# In[ ]:


torch.save(model.state_dict(), "/workspace/llama3.2-1b-sae/sae.pth")


# In[ ]:


sae2 = Sae(d_in, d_sae)
sae2.load_state_dict(model.state_dict())


# In[ ]:



