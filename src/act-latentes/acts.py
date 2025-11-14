"""
Gets the activations from a llama mlp from a HuggingFace dataset,
Passes them to the encoder of an SAE trained on that layer
Uses a running topk to get the locations in the dataset of maximal
latent value
"""


# History of fails:
# 1. We had windows of variable length in numpy, we were sending a list of such
#    numpy object out of collate, this was a massive bottleneck
# 2. We tried padding/unpadding just to concatenate them for bigger encoder matmuls
#    Profiling then showed the main thread burning ~75 % of its time serializing
#    activation tensors to JSON.
# 3. Manual shared memory (not .share_memory_() nor automatic DataLoader shared
#    mem), we experimented with storing data in the format
#    [batch_meta][dim_meta][window_sizes_data][latent_data][text_data]
# 4. We split latent computation from (de)tokenization
# 5. The latent side moved to Zarr output; we even tested zstd vs gzip, but
#    gzip never beat zstd in size.
# 6. Then we switched to fixed-length windows

from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import hf_hub_download
import torch
from torch import Tensor
import numpy as np
from tqdm import tqdm
import uuid
from collections.abc import Callable
import os
from dataclasses import dataclass


N_TOP_K = 200  # Number of top activations to track per dimension
# 8 is higher than on the SAE arena study guide (which used 3 to 5) but that
# guide is no field handbook
BUFFER_SIZE = 8  # Spacing buffer to avoid selecting nearby tokens

EXPECTED_NORM = 3.4  # Expected norm of the activation vectors


@dataclass
class SAEParams:
    """Parameters for the Sparse Autoencoder model."""
    enc_w:     Tensor  # [d_sae, d_model]
    enc_b:     Tensor  # [d_sae]
    dec_b:     Tensor  # [d_model]
    exp_norm:  float
    threshold: Tensor  # [d_sae]


# If One uses a plain running top-k, then the results end up
# all clustering sequentially, even one after another, thus this is
# important, as for autointerp we don't want repeated text windows
# (we will analyze text windows around the tokens of max activations of the
# dataset)

# Directly inspired from a function in the SAE arena
# After testing that the function in the SAE arena was
# not as performant as initially thought, so we switch to the following approach
def topk_with_spacing(
    x:      Tensor,  # [seq, d_sae]
    k:      int,
    buffer: int,
) -> tuple[Tensor, Tensor]:  # [k, d_sae], [k, d_sae]
    """
    An alternative to torch.topk(x, dim=0) with spacing constraints.

    Selects top-k values per column (dimension) while ensuring selected indices
    are spaced at least `buffer` positions apart within each column.

    Args:
        x: Input tensor of shape [seq, d_sae] where seq is sequence length (batch tokens)
        k: Number of top values to select per dimension (column)
        buffer: Minimum spacing between selected indices (e.g., buffer=8 means
                selected positions must be >8 apart from each other)

    Returns:
        indices: Tensor of shape [k, d_sae] containing selected sequence positions per dimension
        values: Tensor of shape [k, d_sae] containing values at those positions

    Example:
        >>> x = torch.tensor([[9, 9],
        ...                   [8, 1],
        ...                   [1, 8]])  # [3, 2]
        >>> indices, values = topk_with_spacing(x, k=2, buffer=1)
        >>> indices
        tensor([[0, 0],
                [2, 2]])
        >>> values
        tensor([[9, 9],
                [1, 8]])
        >>> # Contrast with torch.topk(x, k=2, dim=0) which gives:
        >>> #   indices=[[0, 0], [1, 2]], values=[[9, 9], [8, 8]]
        >>> # Column 0: buffer version selects [0, 2] instead of [0, 1] because positions
        >>> #            0 and 1 are only 1 apart (violates buffer=1), so picks position 2
        >>> # Column 1: buffer version matches topk [0, 2] because positions 0 and 2 are
        >>> #            already 2 apart (satisfies buffer=1)
    """
    # tldr: top-k on more elements than needed, then discard to avoid overlap,
    # it repeats the top-k if needed
    assert x.dim() == 2, "Expected a 2D tensor"
    seq_len, d_sae = x.shape
    assert buffer >= 0, "buffer must be non-negative"
    max_selectable = (seq_len + buffer) // (buffer + 1)
    assert k <= max_selectable, (
        f"Cannot pick {k} indices with spacing>{buffer}; maximum possible is {max_selectable}"
    )
    assert k > 0

    # Get more than k candidates as we will remove some
    # If there aren't enough elements then we to another topk
    candidate_multiplier = 2 * buffer + 1
    candidate_count = min(seq_len, k * candidate_multiplier)

    candidate_values, candidate_indices = torch.topk(
        x,
        candidate_count,
        dim     = 0,  # Column-wise: find top-k per dimension
        largest = True,
        sorted  = True,
    )
    # Now: candidate_values and candidate_indices are [candidate_count, d_sae]

    selected_indices = torch.full(
        (k, d_sae),
        fill_value = -1,
        dtype      = torch.long,
        device     = x.device,
    )
    selected_values = x.new_zeros(k, d_sae)
    counts = x.new_zeros(d_sae, dtype=torch.long)
    arange_dims = torch.arange(d_sae, device=x.device)

    for rank in range(candidate_count):
        idx_candidate = candidate_indices[rank, :]  # [d_sae]
        val_candidate = candidate_values[rank, :]   # [d_sae]
        available = counts < k

        if available.any():
            candidate_dims = arange_dims[available]
            candidate_idx = idx_candidate[available]
            candidate_val = val_candidate[available]

            if buffer > 0:
                prev_selected = selected_indices[:, candidate_dims]  # [k, n_available]
                diffs = prev_selected - candidate_idx.unsqueeze(0)  # Broadcast along k
                conflict = (prev_selected >= 0) & (diffs.abs() <= buffer)
                keep_mask = ~conflict.any(dim=0)  # Check along k dimension
            else:
                keep_mask = torch.ones(candidate_dims.size(0), dtype=torch.bool, device=x.device)

            if keep_mask.any():
                dims = candidate_dims[keep_mask]
                rows = counts[dims]  # Which row (k position) to fill
                selected_indices[rows, dims] = candidate_idx[keep_mask]
                selected_values[rows, dims] = candidate_val[keep_mask]
                counts[dims] = counts[dims] + 1

        if torch.all(counts >= k):
            break

    needs_fill = counts < k
    if needs_fill.any():
        fill_dims = arange_dims[needs_fill]
        for dim in fill_dims.tolist():
            remaining = k - counts[dim].item()
            if remaining <= 0:
                continue
            vals, idxs = torch.topk(x[:, dim], k, dim=0, largest=True, sorted=False)
            take = min(remaining, idxs.size(0))
            start = counts[dim].item()
            selected_indices[start:start + take, dim] = idxs[:take]
            selected_values[start:start + take, dim] = vals[:take]
            counts[dim] += take

    return selected_indices, selected_values

def optimized_numpy_collate(
        batch: list[np.ndarray]  # List of samples, each [d_model]
) -> Tensor:  # [batch_size, d_model] where batch_size = seq (number of tokens)
    """
    Optimized collate function for raw HF dataset activations.

    Note: Each sample is a single token's activation vector [d_model].
    The batch_size dimension represents multiple tokens (i.e., seq length).

    VERY IMPORTANT:
    - We send torch tensors out of collate, they get passed through shared
      memory automatically, we measure a ~2,500x speedup by doing this vs
      sending numpy data
    - We do the ops that take advantage of batched computation here, yielding a
      noticeable perf improvement
    """
    # Stack raw activations: [batch_size, d_model] where batch_size = seq (token count)
    stacked_np = np.stack(batch, axis=0)

    # Convert int64 -> uint16 -> bfloat16 for SAE processing (same as other scripts)
    activations_uint16 = stacked_np.astype(np.uint16)
    activations_tensor = (torch.from_numpy(activations_uint16)
                         .view(torch.bfloat16))

    return activations_tensor

def get_params() -> SAEParams:
    sae = hf_hub_download(
        repo_id="mech-interp-uam/llama3.2-1b-sae",
        filename="sae_exp24_sparse0.001_d_sae_std_fullwarmup_steps256000_lr7e-05.pth",
        repo_type="model",
        revision="main",
    )
    sd = torch.load(sae, map_location="cpu")
    threshold = sd["log_threshold"].exp()
    enc_w: Tensor = sd["enc.weight"]  # [d_sae, d_model]
    enc_b: Tensor = sd["enc.bias"]    # [d_sae]
    dec_b: Tensor = sd["dec.bias"]    # [d_model]
    exp_norm = EXPECTED_NORM
    return SAEParams(
        enc_w=enc_w,
        enc_b=enc_b,
        dec_b=dec_b,
        exp_norm=exp_norm,
        threshold=threshold,
    )


# The dataset is not correctly configured, we have to download it this way
# TODO: is the order guaranteed to be the same each time?
# TODO: guarantee the order
def get_ds():  # type: ignore[no-untyped-def]
    dataset_id = "naraca/activaciones-llama3-mlp8"
    dataset_dict = load_dataset(dataset_id)
    splits = [
        split for split in dataset_dict.keys() if split.startswith("train_")
    ]
    print(f"{len(splits)}")
    ds = concatenate_datasets([dataset_dict[split] for split in splits])
    print(f"Dataset loaded successfully: {len(ds)=}")
    return ds

def compute_latents_closure(
    enc_w: Tensor,      # [d_sae, d_model]
    enc_b: Tensor,      # [d_sae]
    dec_b: Tensor,      # [d_model]
    exp_norm: float,
    threshold: Tensor,  # [d_sae]
) -> Callable[[Tensor], Tensor]:  # [seq, d_model] -> [seq, d_sae]
    @torch.compile
    # Increase this counter each time you
    # forget to divide by the expected norm: 2
    def compute_latents(x: Tensor) -> Tensor:  # x: [seq, d_model] -> [seq, d_sae]
        pre_acts =  (x/exp_norm - dec_b)@enc_w.T + enc_b
        acts = (pre_acts > threshold) * pre_acts
        return acts
    return compute_latents

def compute_pre_latents_closure(
    enc_w: Tensor,       # [d_sae, d_model]
    enc_b: Tensor,       # [d_sae]
    dec_b: Tensor,       # [d_model]
    exp_norm: float = EXPECTED_NORM,
) -> Callable[[Tensor], Tensor]:  # [seq, d_model] -> [seq, d_sae]
    @torch.compile
    def compute_pre_latents(x: Tensor) -> Tensor:  # x: [seq, d_model] -> [seq, d_sae]
        pre_acts = (x/exp_norm - dec_b)@enc_w.T + enc_b
        return pre_acts
    return compute_pre_latents

class ActivationsDataset(Dataset):
    def __init__(self, hf_dataset):
        self.data = hf_dataset

    def __getitem__(self, idx):
        # Return raw activations data - all conversion happens in collate function
        return self.data[idx]['activacion']

    def __len__(self):
        return len(self.data)

def get_stats(
    enc_w:      Tensor,  # [d_sae, d_model]
    enc_b:      Tensor,  # [d_sae]
    dec_b:      Tensor,  # [d_model]
    dataloader: DataLoader,  # type: ignore[type-arg]
    device:     str,
    compute_pre_latents: Callable[[Tensor], Tensor],  # [seq, d_model] -> [seq, d_sae]
    k:             int  = N_TOP_K,
    buffer:        int  = BUFFER_SIZE,
    show_progress: bool = True,
    early_break:   int | None = None,
) -> tuple[Tensor, Tensor]:  # Returns ([k, d_sae], [k, d_sae])
    # Initialize
    global_offset = 0
    global_idx:     Tensor | None = None  # [k, d_sae]
    global_latents: Tensor | None = None  # [k, d_sae]

    total_samples = len(dataloader.dataset)
    pbar = tqdm(total=total_samples, unit='samples', desc='Processing', disable=not show_progress)

    for i, batch_activations in enumerate(dataloader):
        # Early break for testing (useful when not on GPU server)
        if early_break is not None and i >= early_break:
            print(f"Early break at batch {i}")
            break

        activations: Tensor = batch_activations.to(device, non_blocking=True)  # [seq, d_model]
        seq = activations.size(0)  # seq = number of tokens in this batch

        # Use compiled function to compute pre-latents (faster, no activation function)
        pre_latents: Tensor = compute_pre_latents(activations)  # [seq, d_sae]

        # Find top-k samples for each SAE dimension using diverse selection
        current_latents: Tensor = torch.zeros(k, pre_latents.size(1), device=device)  # [k, d_sae]
        current_idx: Tensor = torch.zeros(k, pre_latents.size(1), device=device, dtype=torch.long)  # [k, d_sae]

        topk_count = min(k, pre_latents.size(0))
        if topk_count > 0:
            indices, values = topk_with_spacing(
                pre_latents,  # [seq, d_sae]
                topk_count,
                buffer=buffer,
            ) # [k, d_sae], [k, d_sae]
            current_idx[:indices.size(0), :] = indices
            current_latents[:values.size(0), :] = values

        # Add global offset to get absolute sample indices (position in full dataset)
        current_idx_global = current_idx + global_offset

        if global_idx is None:
            # initialize
            global_idx = current_idx_global
            global_latents = current_latents
        else:
            # Fuse global and local
            combined_idx:     Tensor = torch.cat([global_idx, current_idx_global],  dim=0)  # [2*k, d_sae]
            combined_latents: Tensor = torch.cat([global_latents, current_latents], dim=0)  # [2*k, d_sae]

            global_latents, idx = torch.topk(combined_latents, k, largest=True, sorted=False, dim=0)
            global_idx = torch.gather(combined_idx, 0, idx)

        global_offset += seq  # Track cumulative position across all batches
        pbar.update(seq)

    pbar.close()
    return global_idx, global_latents


def run_acts(
    k:             int = N_TOP_K,
    buffer:        int = BUFFER_SIZE,
    dim_range:     tuple[int, int] | None = None,
    show_progress: bool = True,
    early_break:   int | None = None,
    output_dir:    str = "top_activations"
) -> dict:
    """
    Run the full acts pipeline to find top-k activations per SAE dimension.

    Args:
        k: Number of top activations to track per dimension
        buffer: Spacing buffer to avoid selecting nearby tokens
        dim_range: Optional tuple (start, end) for dimension range
        show_progress: Whether to show progress bar
        early_break: Optional number of batches to process before stopping
        output_dir: Directory to save output files

    Returns:
        dict with keys: latents_path, indices_path, run_id
    """

    params = get_params()
    enc_w     = params.enc_w
    enc_b     = params.enc_b
    dec_b     = params.dec_b
    exp_norm  = params.exp_norm
    threshold = params.threshold
    ds = get_ds()
    ds.set_format("numpy")  # Set format for optimized collate function

    # Get dataset name for output filename
    dataset_id = "naraca/activaciones-llama3-mlp8"
    dataset_short = dataset_id.split('/')[-1].replace('activaciones-', '').replace('-', '_')

    # Setup device and move tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        # Optimized based on custom collate scaling analysis:
        # 6-8 workers give best efficiency/performance balance
        # Custom collate eliminates the >20 workers performance degradation
        batch=1024*8*6  # 6x batch size: 49152 (conservative to avoid OOM)
        prefetch_factor=2
        pin_memory=True
        num_workers=16  # 2x workers (from successful run)
    else:
        batch=1024*8
        prefetch_factor=2
        pin_memory=False
        num_workers=8

    # Filter dimensions if specified
    if dim_range is not None:
        start, end = dim_range
        if start > end:
            raise ValueError("START must be less than or equal to END for --dim-range.")
        if start < 0 or end >= enc_w.shape[0]:
            raise ValueError(f"Dimension range [{start}, {end}] out of bounds for SAE with {enc_w.shape[0]} dimensions.")

        print(f"Filtering to dimensions: {start}-{end} ({end - start + 1} total)")
        enc_w = enc_w[start:end+1]
        enc_b = enc_b[start:end+1]
        threshold = threshold[start:end+1]
        dim_range_str = f"_dims{start}-{end}"
    else:
        dim_range_str = ""

    enc_w = enc_w.to(device)
    enc_b = enc_b.to(device)
    dec_b = dec_b.to(device)
    threshold = threshold.to(device)

    # Create compiled functions
    compute_pre_latents = compute_pre_latents_closure(enc_w, enc_b, dec_b, exp_norm)

    # Create dataloader
    dataloader = DataLoader(
        ActivationsDataset(ds),
        batch_size=batch,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=False,
        drop_last=True,
        collate_fn=optimized_numpy_collate,  # Use optimized collate for ~3.7x speedup
    )

    # Run the thing
    top_indices, top_latents = get_stats(
        enc_w,
        enc_b,
        dec_b,
        dataloader,
        device,
        compute_pre_latents,
        k=k,
        buffer=buffer,
        show_progress=show_progress,
        early_break=early_break,
    )


    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build descriptive filename
    run_id = uuid.uuid4().hex[:8]
    d_sae = enc_w.shape[0]
    d_model = enc_w.shape[1]
    device_str = device if device != 'cuda' else 'gpu'
    filename_base = f"top_from_{dataset_short}_k{k}_buf{buffer}_dsae{d_sae}_dmodel{d_model}{dim_range_str}_{device_str}_{run_id}"

    latents_path = os.path.join(output_dir, f"{filename_base}_latents.npy")
    indices_path = os.path.join(output_dir, f"{filename_base}_indices.npy")

    np.save(latents_path, top_latents.to("cpu").numpy())
    np.save(indices_path, top_indices.to("cpu").numpy())

    print(f"\nSaved results:")
    print(f"  Latents: {latents_path}")
    print(f"  Indices: {indices_path}")

    return {
        "latents_path": latents_path,
        "indices_path": indices_path,
        "run_id": run_id
    }


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(
        description="Find top-k activations per SAE dimension from dataset."
    )
    parser.add_argument(
        "--k",
        type=int,
        default=N_TOP_K,
        help=f"Number of top activations to track per dimension (default: {N_TOP_K}).",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=BUFFER_SIZE,
        help=f"Spacing buffer to avoid selecting nearby tokens (default: {BUFFER_SIZE}).",
    )
    parser.add_argument(
        "--dim-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive range of SAE dimensions to process. Omit to process all dimensions.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the tqdm progress bar.",
    )
    parser.add_argument(
        "--early-break",
        type=int,
        metavar="N",
        help="Stop after N batches (for testing when not on GPU server).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Convert args to function parameters
    dim_range = tuple(args.dim_range) if args.dim_range else None

    # Call run_acts with parameters from CLI
    result = run_acts(
        k=args.k,
        buffer=args.buffer,
        dim_range=dim_range,
        show_progress=not args.no_progress_bar,
        early_break=args.early_break,
    )

if __name__ == "__main__":
    main()
