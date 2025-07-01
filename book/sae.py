import torch
from torch import nn
import jaxtyping
import dataclasses
from tqdm import tqdm
from torch.nn import functional as F
from torch.linalg import vector_norm
import math
import datasets
import numpy as np
from torch.utils.data import Dataset, DataLoader

dtype = torch.float32
device = "cuda" if torch.cuda.is_available else "cpu"

class Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        ctx.save_for_backward(x, threshold)
        return (x > threshold).to(x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        bandwidth = 0.001
        x, threshold = ctx.saved_tensors

        grad_threshold = torch.where(
            abs(F.relu(x) - threshold) < bandwidth/2,
            -1.0/bandwidth, 0)
        
        return torch.zeros_like(x), grad_threshold * grad_output

class Sae(nn.Module):
    def __init__(self, d_in, d_sae, use_pre_enc_bias=True, **kwargs):
        super().__init__(**kwargs)
        self.enc = nn.Linear(d_in, d_sae, dtype=dtype)
        self.dec = nn.Linear(d_sae, d_in, dtype=dtype)
        with torch.no_grad():
            # normalize each of the d_sae dictonary vectors
            self.dec.weight /= vector_norm(self.dec.weight, dim=0, keepdim=True)
            self.enc.weight.copy_(self.dec.weight.clone().t())
            self.enc.bias.copy_(torch.zeros_like(self.enc.bias))
            self.dec.bias.copy_(torch.zeros_like(self.dec.bias))
        self.log_threshold = nn.Parameter(
            torch.log(torch.full((d_sae,), 0.001, dtype=dtype)))
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
            project_out_parallel_grad(0, self.dec.weight))
                

    def forward(self,
        x,
        return_mask=False,
        return_l0=True,
        return_reconstruction_loss=True,
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
        if return_mask:
            d['mask'] = s
        if return_l0:
            d['l0'] = s.float().mean(0).sum(-1)
        if not return_reconstruction_loss:
            return d
        x = x*s
        x = self.dec(x)

        d['reconstruction'] = ((x - original_input).pow(2)).mean(0).sum()

        return d


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

def sparsity_schedule(step, warmup_steps, max_sparsity_coeff):
    if step >= warmup_steps:
        return max_sparsity_coeff
    return max_sparsity_coeff*((step+1) / warmup_steps)

class ActivationsDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __getitem__(self, idx):
        # Return as float16, on modern GPUs conversion from float 16 to 32 is
        # free compared to matmults or so I was told
        return torch.tensor(self.data[idx]['activations'])

    def __len__(self):
        return len(self.data)

def train():
    # Load the dataset
    ds = datasets.load_dataset('mech-interp-uam/llama-mlp8-outputs')
    ds.set_format('numpy')

    # Check the dataset structure
    print(f"Dataset loaded successfully: {len(ds['train'])} examples")
    print(f"Features: {ds['train'].features}")
    print(f"First example shape: {ds['train'][0]['activations'].shape}")


    batch = 1024
    dataset = ActivationsDataset(ds['train'])
    dataloader = DataLoader(dataset, batch_size=batch, shuffle=True, num_workers=4)


    steps = 2**16
    max_lr = 7e-5
    d_in = 2048
    d_sae = 2048*8
    model = Sae(d_in, d_sae)
    model.to('cuda')
    #model.compile()
    warmup_steps=2000
    sparcity_warmup_steps=10000
    total_steps=32000 #for now
    optimizer = torch.optim.Adam([
        {"params": model.enc.parameters(), "lr":max_lr, "betas":(0.0,0.999)},
        {"params": model.dec.parameters(), "lr":max_lr, "betas":(0.0,0.999)},
        {"params": model.log_threshold, "lr":3.5*max_lr, "betas":(0.9,0.999)},
    ])
    max_sparsity_coeff = 0.0005

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_schedule_with_warmup(step, warmup_steps, total_steps)
    )
    # train_ds = fake_train_loader(batch, d_in, total_steps)
    total_step = 0
    should_break = False
    for epoch in range(1000):
        for step, x in enumerate(tqdm(dataloader)):
            x /= 3.4 # this is supposed to be the expected norm
            x = x.to(dtype).to("cuda")
            optimizer.zero_grad()
            d = model(x)
            reconstruction_loss, l0 = d['reconstruction'], d['l0']
            sparsity_coefficient = sparsity_schedule(total_step, sparcity_warmup_steps, max_sparsity_coeff)
            loss = reconstruction_loss + sparsity_coefficient * l0
            # log losses, compute stats, etc
            grad = loss.backward()
            # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # metrics
            if (total_step % 500) == 0:
                with torch.no_grad():
                    # print metrics
                    print(f"reconstruction={reconstruction_loss.item()}")
                    print(f"l0={l0.item()}")
                    # print(f"norm={norm.item()}")
                    print(f"{sparsity_coefficient=}")
            optimizer.step()
            # TODO: sparsity_coefficient scheduler
            # print(scheduler.get_lr())
            scheduler.step()

            # normalize
            with torch.no_grad():
                wdecnorm = vector_norm(model.dec.weight, dim=0, keepdim=True)
                model.dec.weight /= wdecnorm
        # print(f"epoch loss: {loss.detach().item()}")
            total_step +=1
            if total_step > total_steps:
                should_break = True
                break
        if should_break:
            break

    torch.save(model.state_dict(), "/workspace/llama3.2-1b-sae/sae.pth")

