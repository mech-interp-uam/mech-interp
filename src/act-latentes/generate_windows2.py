# Ok, after a lot of experimentation and ideas that were proven suboptimal, here
# Background:
# we have the middle state of running a llama model, the outputs from the mlp 8
# while ran on a corpus similar to those used during pretraining
# using that, an sparse autoencoder was trained, we collected the topk locations
# of max value in the data for each dim of the latent (after the
# autoencoder finished training).

# Now, we want to visit those locations and compute a window of latent vectors
# (the model + SAE generate one latent vector per token of text)
# No need to run the model by itself, we have the middle activations already
# stored

# An finally, no need to save all activations, given a window of acts around the
# location of max value of dim index k, we are interested on only the k
# component of the latent vector at each location of the window

# No need to mix with tokenization, we can tokenize in another part of the code

# We need to think how to go about these matmuls
# esentially, each activation is going to get map to a scalar
# we do this for all activations in a window for all windows in dim
# for all dims in sae

# what if we pad to get our tensors
# data:
# d_sae n_w t d_model
# enc_w:
# d_sae d_model:

# we essentially want our output to be:
# d_sae n_w t

# Cijk = sum_l Aijkl Bil

# This is not a matmul, matmuls reduce stuff by
# one dimension

# we could batch over d_sae if needed

# After some maths, we found that the window locations are quite close, and thus
# each data worker ends up loading the entire dataset to ram because the access
# are random, running on the server we found that the main thread had an average
# useage of 20% with 12 dataworkers, and the GPU stayed at 1%, the server has a
# lot of cores but RAM is the current limiting factor, and thus we need to
# *SORT OUR WINDOWS*, do the matmuls, and return to the original order

# Imagine a grid of d_sae by n_w, each square in the grid contains a sequence of
# activation vectors. Then, imagine the another grig of the same shape, but
# this time each square contains a single vector. What we do is "multiply" both
# grid square-by square: in a single square of the former we have a seq of
# vectors, of the later we got a single vector, we multiply each vec in the seq
# to get an scalar, repeating this to finally get a *SEQUENCE OF VECTORS* which
# we place in the corresponding square of the output grid

# Now, we can apply a permutation to both of them (the same permutation to both)
# we do the matmul, and apply the reverse permutation, this resuls in the same
# output as if we didn't permute/unpermute

# There is a detail: to actually do stuff this way, we could sort the window
# locations in advance, but the encoder weights

# We use the previous fact to order our windows. but consider that we might use
# the same vector of w_dec multiple times or even not at all each time we do a
# matmul using a subset of the data (essentially, each window has it's
# associated sae dim, we use the vector for that dim)

# So, each batch we should get a vector of data:
# b n_w t d_model
# notice here we have subset the first dim
# but also a vector of the permutation of the squares in the original 
# (b, n_w) grid (thus, an int tensor of 2 axis)
# This int tensor has the rows we want from the encoder:
# w_enc[permutation], permutation.shape = (b, n_w),
# w_enc.shape = (d_sae, n_w, t, d_model)
# w_enc[permutation].shape = (b, n_w, t, d_model)

# Then we multiply as such:
# acts_sorted * w_enc[permutation]

# Can this be done with scatter?
# Each batch the stuff from W_enc we use is different
# It seems this is a gather problem instead of a gather one,
# and that index_select_ is about the same as w_enc[perm]
# in terms of the ops that happen internally

# If we sort the windows, then we can even work with tensors of shape
# (n_w', t, d_model)
# where we have included multiple windows from possibly multiple dimensions of
# the sae in the 0th axis n_w'
# but then the dataloader should also give us a tensor of shape
# (n_w')
# of the matching dims
# and the output would be
# (n_w', t)


# independent of if we sort, it is easy to check if a result is close to the
# edge, what about the windows we return instead of checking and padding all of
# them, we make it fixed len by the last and first windows on the edge of the
# dataset we move them, they will still contain the token of high latent value
# but this way the main thread would do less work

# To know wich activation latent is a asociated with, each latent, we need:
# the dataloader to emmit it. Given a sequence s = [i_1, ...,i_n] of the
# w[i,:] we want of the sae we could get that with w[s,:] == w[s[1_i,... , i_n]].

# Ok, for storing the data, we need to know not only the latent, but also the
# window within that latent, or alternatively the position within a
# flatten (d_sae, n_w) array.

# We could actually send this ahead of time to the zarr file, as a sidecar, that
# is, given a fixed order for the outgoing data from DataLoader, we could
# replicate that order ahead of time and send the windo number within the latent
# or the number on the flatten (d_sae, n_w) array. This would give us some
# performance improvement. We leave the implementation as an exercise to the
# reader. That is, to confirm how much gains this give us

import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, Subset
from huggingface_hub import hf_hub_download
from acts import get_ds
import zarr

WINDOW_LEN = 64
EXPECTED_NORM = 3.4  # Expected norm of the activation vectors


class DimensionWiseDataset(Dataset):
    def __init__(
        self,
        # sorted_indices, _ = indices.flatten().sort()
        sorted_indices: torch.Tensor,
        original_ds,
        # _, perm = indices.flatten().sort()
        perm: torch.Tensor,
        n_windows_per_dim: int,
        original_dim_ids: list[int],  # Maps filtered dim index -> original SAE dim ID
        window_len: int = WINDOW_LEN,
    ):
        self.sorted_indices = sorted_indices
        self.original_ds = original_ds
        self.window_len = window_len

        # Precompute mapping: sorted_position -> SAE dimension ID
        # This allows us to return the SAE dim ID directly from __getitem__
        original_flat_indices = perm  # original_flat_indices[i] is the original flat index for sorted position i
        filtered_dim_indices = original_flat_indices // n_windows_per_dim
        self.sae_dim_ids = torch.tensor([original_dim_ids[i] for i in filtered_dim_indices.tolist()],
                                        dtype=torch.int32)

    def __len__(self):
        return self.sorted_indices.shape[0]

    def __getitem__(self, idx: int):
        """
        Get a single window of llama mlp activations.
        Returns: (sorted_pos, sae_dim_id, activation_data)
          - sorted_pos: position in sorted order (for writing to zarr)
          - sae_dim_id: SAE dimension ID (for directly indexing: enc_w[sae_dim_id])
        """
        # Get the sorted data index
        data_idx = int(self.sorted_indices[idx].item())

        # Calculate window boundaries with FIXED window size
        start_idx = data_idx - self.window_len // 2
        end_idx = start_idx + self.window_len

        if start_idx < 0:
            start_idx = 0
            end_idx = self.window_len
        elif end_idx > len(self.original_ds):
            end_idx = len(self.original_ds)
            start_idx = end_idx - self.window_len

        # Load activation window
        activation_vector_window = self.original_ds[start_idx:end_idx]["activacion"]

        # Return: (sorted_pos, sae_dim_id, activation_data)
        return idx, self.sae_dim_ids[idx].item(), activation_vector_window

def writer_worker(
    queue,
    store_path: str,
    total_windows: int,
    seq_len: int,
    storage_kind: str,
):
    """
    Worker process that writes queued batches of tensors into a Zarr store (in sorted order).

    REQUIREMENTS: Python 3.11+ and zarr 3.1.3+
    - zarr 3.0.0a5 (last version for Python 3.10) has a multiprocessing bug that causes hangs
    - zarr 3.1.3+ requires Python 3.11+
    - Use pytorch/pytorch:2.5.1+ Docker images which include Python 3.11
    """

    # IMPORTANT: Use mode="r+" not "w" to preserve the permutation/metadata
    # that was already saved by the main process. Using "w" would wipe it!
    root = zarr.open_group(store_path, mode="r+")
    latents_ds = root.create_array(
        "latents_sorted",
        shape=(total_windows, seq_len),
        chunks=(1000, seq_len),  # Chunk by windows instead of dims
        dtype=np.uint16,
    )
    # No need to store lengths - all windows are fixed size (seq_len)
    # Add additional attrs (main process already set n_windows_per_dim and d_sae)
    root.attrs["storage_kind"] = storage_kind
    root.attrs["seq_len"] = seq_len
    root.attrs["total_windows"] = total_windows

    while True:
        item = queue.get()

        # Sentinel to stop
        if item is None:
            break

        # Now receiving batches: (flat_indices, latents_batch)
        # lens_batch removed - all windows are fixed length
        flat_indices, latents_batch = item

        # Convert to numpy for zarr - zarr expects numpy arrays for fancy indexing
        flat_indices_np = flat_indices.numpy()
        latents_uint16 = latents_batch.view(torch.uint16).numpy()

        # CRITICAL: Use fancy indexing to write entire batch at once!
        # Writing one window at a time (loop with latents_ds[i, :] = ...) would
        # do 10M zarr writes and become a massive bottleneck.
        latents_ds[flat_indices_np, :] = latents_uint16

def generate_dim_data(
    dataloader,
    dec_b,
    enc_w,
    enc_b,
    threshold,
    n_windows_per_dim,
    original_dim_ids,  # Maps from filtered dim index to original SAE dim (for logging/info only)
    device,
    store_path: str,
    seq_len: int,
    compute_dtype: torch.dtype = torch.float32,
    show_progress: bool = True,
):
    import torch.multiprocessing as mp

    if zarr is None:
        raise RuntimeError(
            "generate_windows2 requires the 'zarr' package for output serialization. "
            "Install it with `pip install zarr numcodecs`."
        )

    total_windows = len(dataloader.dataset)

    storage_kind = "bfloat16"
    storage_torch_dtype = torch.bfloat16

    # Normalization constant from acts.py
    exp_norm = 3.4

    # Define computation kernel
    def compute_latents_batch(acts, dec_b_local, enc_w_local, enc_b_local, threshold_local):
        """
        Computation kernel for batch of latents.
        Formula from acts.py: pre_acts = (x/exp_norm - dec_b) @ enc_w.T + enc_b
        """
        # Normalize and center activations
        normalized = acts / exp_norm
        centered = normalized - dec_b_local
        # Compute pre-activations: (batch, seq, d_model) @ (batch, d_model) -> (batch, seq)
        pre_latents = torch.einsum('btm,bm->bt', centered, enc_w_local) + enc_b_local.view(-1, 1)
        # Apply threshold (ReLU with threshold)
        return pre_latents * (pre_latents > threshold_local.view(-1, 1))

    # Compile for GPU only - significant speedup by JIT-compiling into optimized CUDA kernels
    if device == 'cuda':
        compute_latents_batch = torch.compile(compute_latents_batch, mode="max-autotune")
        print(f"Compiling computation kernel (first batch will be slow)...")
    else:
        print(f"Running on CPU - compilation disabled")

    # Start writer process with torch multiprocessing queue
    queue = mp.Queue(maxsize=50)  # Buffer up to 50 batches
    writer_process = mp.Process(
        target=writer_worker,
        args=(queue, store_path, total_windows, seq_len, storage_kind),
    )
    writer_process.start()

    progress_bar = tqdm(total=total_windows, desc="Windows", leave=False) if show_progress else None
    log_interval = max(1, total_windows // 10) if total_windows else 1

    processed_windows = 0

    # Batch accumulator for writing - accumulate ~200 windows before sending.
    # This reduces queue overhead: instead of many small queue.put() calls,
    # we do fewer larger calls, while keeping memory bounded.
    WRITE_BATCH_SIZE = 200
    write_batch_indices = []  # Accumulate on CPU
    write_batch_latents = []  # Accumulate on GPU

    try:
        iterator = dataloader
        for sorted_positions, sae_dim_ids, all_acts_tensor in iterator:
            batch_size = len(sorted_positions)

            # Move to device (sorted_positions stays on CPU - no need to move it!)
            sae_dim_ids = sae_dim_ids.to(device, non_blocking=True)
            all_acts_tensor = all_acts_tensor.to(device, dtype=compute_dtype, non_blocking=True)
            # Note: all_acts_tensor is already reshaped by collate function

            # Gather encoder weights using precomputed SAE dimension IDs
            # sae_dim_ids come directly from the dataset, already mapped to original SAE dimensions
            enc_w_batch = enc_w[sae_dim_ids]  # (batch_size, d_model)
            enc_b_batch = enc_b[sae_dim_ids]  # (batch_size,)
            threshold_batch = threshold[sae_dim_ids]  # (batch_size,)

            # Compute latents using compiled kernel for maximum GPU utilization
            latents = compute_latents_batch(
                all_acts_tensor, dec_b, enc_w_batch, enc_b_batch, threshold_batch
            ).clone()  # clone() breaks CUDAGraph output aliasing across steps

            # Accumulate on GPU (no immediate transfer to CPU)
            write_batch_indices.append(sorted_positions)  # Already on CPU
            write_batch_latents.append(latents)  # Keep on GPU

            # Send batch when it reaches target size
            if sum(len(idx) for idx in write_batch_indices) >= WRITE_BATCH_SIZE:
                # Concatenate on GPU (fast)
                batch_latents_gpu = torch.cat(write_batch_latents)

                # One large GPU->CPU transfer (more efficient than many small ones)
                batch_latents_cpu = batch_latents_gpu.to("cpu", dtype=storage_torch_dtype, non_blocking=True)

                # Concatenate positions on CPU
                batch_indices_cpu = torch.cat(write_batch_indices)

                # CRITICAL: share_memory_() avoids pickling the tensor data when
                # passing through queue. Only metadata gets serialized, saving
                # massive overhead.
                batch_indices_cpu.share_memory_()
                batch_latents_cpu.share_memory_()

                queue.put((batch_indices_cpu, batch_latents_cpu))

                # Clear the accumulators
                write_batch_indices.clear()
                write_batch_latents.clear()

            processed_windows += batch_size
            if progress_bar:
                progress_bar.update(batch_size)
            else:
                if processed_windows == batch_size or processed_windows % log_interval == 0 or processed_windows == total_windows:
                    print(f"Processed {processed_windows}/{total_windows} windows")

        # Send any remaining windows in the batch
        if write_batch_indices:
            # Concatenate on GPU then transfer
            batch_latents_gpu = torch.cat(write_batch_latents)
            batch_latents_cpu = batch_latents_gpu.to("cpu", dtype=storage_torch_dtype, non_blocking=True)

            # Concatenate positions on CPU
            batch_indices_cpu = torch.cat(write_batch_indices)

            batch_indices_cpu.share_memory_()
            batch_latents_cpu.share_memory_()
            queue.put((batch_indices_cpu, batch_latents_cpu))

    finally:
        # Send sentinel to stop writer
        queue.put(None)
        writer_process.join()
        if progress_bar:
            progress_bar.close()
        print("Writer process finished")

def dim_collate(
    batch: list[tuple[int, int, np.ndarray]],
):
    """
    Collate function for sorted windows.
    batch: list of (sorted_pos, sae_dim_id, activation_window)

    All windows are guaranteed to be exactly WINDOW_LEN tokens (no padding needed).
    Reshapes activations here in worker thread to reduce main thread load.
    """
    # Unzip batch into separate lists
    sorted_positions, sae_dim_ids, acts_list = zip(*batch)

    # Convert to tensors
    sorted_positions = torch.tensor(sorted_positions, dtype=torch.int64)
    sae_dim_ids = torch.tensor(sae_dim_ids, dtype=torch.int64)
    all_acts = np.concatenate(acts_list, axis=0).astype(np.uint16)
    all_acts_tensor = torch.from_numpy(all_acts).view(torch.bfloat16)

    # Reshape here in worker thread: (batch_size * seq_len * d_model) -> (batch_size, seq_len, d_model)
    batch_size = len(sorted_positions)
    seq_len = 64  # WINDOW_LEN
    d_model = 2048
    all_acts_tensor = all_acts_tensor.view(batch_size, seq_len, d_model)

    return sorted_positions, sae_dim_ids, all_acts_tensor



def setup_and_generate(
        dims: list[int],
        latents_file: str,
        indices_file: str,
        show_progress: bool = True,
    ):
    if torch.cuda.is_available():
        device = 'cuda'
        torch.set_float32_matmul_precision('high')
        batch_size=128 # Increased from 16 - 4090 has plenty of VRAM
        prefetch_factor = 4  # Increased from 2 - keep data pipeline full
        pin_memory = True
        num_workers = 20  # Increased from 8 - more parallel data loading
        compute_dtype = torch.bfloat16
    else:
        device = 'cpu'
        batch_size=4
        prefetch_factor = 2
        pin_memory = False
        num_workers = 12
        compute_dtype = torch.float32

    # Get SAE path
    sae_path = hf_hub_download(
        repo_id="mech-interp-uam/llama3.2-1b-sae",
        filename="sae_exp24_sparse0.001_d_sae_std_fullwarmup_steps256000_lr7e-05.pth",
        repo_type="model",
        revision="main",
    )

    # Load SAE parameters and move to device
    # Note: dec_b is needed for the computation: centered = normalized - dec_b
    sd = torch.load(sae_path, map_location=device)
    dec_b = sd["dec.bias"].to(device, dtype=compute_dtype)
    enc_w = sd["enc.weight"].to(device, dtype=compute_dtype)  # (d_sae, d_model)
    enc_b = sd["enc.bias"].to(device, dtype=compute_dtype)  # (d_sae,)
    threshold = sd["log_threshold"].exp().to(device, dtype=compute_dtype)  # (d_sae,)

    # Free memory: delete state dict and decoder weights we don't need
    # (we only use dec_b, not dec.weight which is much larger)
    del sd

    top_indices = np.load(indices_file)
    ds = get_ds()
    ds.set_format('numpy')
    print(f"{top_indices.shape=}")

    # Convert indices to torch tensor
    indices_tensor = torch.from_numpy(top_indices)  # (n_windows, d_sae)
    n_windows_per_dim = indices_tensor.shape[0]
    d_sae_full = indices_tensor.shape[1]

    # Filter dimensions if specified
    if dims is not None:
        print(f"Filtering to dimensions: {dims[0]}-{dims[-1] if len(dims) > 1 else dims[0]} ({len(dims)} total)")
        # Select only the specified dimension columns
        indices_tensor = indices_tensor[:, dims]  # (n_windows, len(dims))
        d_sae = len(dims)
        original_dim_ids = dims  # Track which dimensions we're processing
    else:
        d_sae = d_sae_full
        original_dim_ids = list(range(d_sae_full))  # All dimensions

    print("Sorting windows by data index...")
    # Sort windows by their data location to enable sequential disk reads.
    # This dramatically reduces RAM usage on the server: workers were loading
    # the entire dataset into RAM due to random access patterns.
    # Flatten: (n_windows, d_sae) -> (n_windows * d_sae,)
    # TODO: does this .T make a difference? we are sorting afterwards
    indices_flat = indices_tensor.T.flatten()  # Transpose then flatten
    sorted_indices, perm = indices_flat.sort()
    # perm[i] tells us: sorted[i] came from original[perm[i]]
    # For unsorting: unsorted[perm[i]] = sorted[i], so we save perm directly

    # Save permutation and metadata BEFORE processing starts.
    # This is critical for crash resilience: if the run crashes, the permutation
    # is already on disk so we can still recover the data structure.
    print("Saving permutation and metadata...")
    import uuid
    import os

    # Set sequence length
    seq_len = WINDOW_LEN

    # Create output directory if it doesn't exist
    output_dir = "generated_windows"
    os.makedirs(output_dir, exist_ok=True)

    # Build descriptive filename with key parameters
    run_id = uuid.uuid4().hex[:8]
    if dims is not None:
        dim_info = f"dims{dims[0]}-{dims[-1]}"
        dim_count = len(dims)
    else:
        dim_info = f"dims0-{d_sae-1}"
        dim_count = d_sae

    # Extract input filename stem (without path and extension) for traceability
    indices_basename = os.path.splitext(os.path.basename(indices_file))[0]
    # Remove common prefixes like "top_indices_" to keep it short
    input_id = indices_basename.replace("top_indices_", "").replace("top_", "")

    # Include: input file ID, number of dimensions, windows per dim, window length, sequence length, dim range, device, dtype, uuid
    device_str = device if device != 'cuda' else 'gpu'
    dtype_str = str(compute_dtype).split('.')[-1]  # torch.bfloat16 -> bfloat16
    filename = f"latents_from_{input_id}_d{dim_count}_nwin{n_windows_per_dim}_wlen{WINDOW_LEN}_seq{seq_len}_{dim_info}_{device_str}_{dtype_str}_{run_id}.zarr"
    output_store = os.path.join(output_dir, filename)

    if zarr is None:
        import zarr as _zarr
        zarr_module = _zarr
    else:
        zarr_module = zarr

    root = zarr_module.open_group(output_store, mode="w")
    perm_array = perm.numpy()
    root.create_array(
        "perm",
        data=perm_array,  # zarr 3.1.3 infers shape and dtype from data
    )
    # Store original dimension IDs (important when using --dim-range).
    # Without this, subset runs would show dims [0,1,2,...] instead of [8,9,10,...].
    original_dim_ids_array = np.array(original_dim_ids, dtype=np.int32)
    root.create_array(
        "original_dim_ids",
        data=original_dim_ids_array,  # zarr 3.1.3 infers shape and dtype from data
    )
    root.attrs["n_windows_per_dim"] = n_windows_per_dim
    root.attrs["d_sae"] = d_sae
    print(f"Permutation saved to {output_store}")

    dataset = DimensionWiseDataset(
        sorted_indices,
        ds,
        perm,
        n_windows_per_dim,
        original_dim_ids,
        window_len=seq_len,
    )

    print(f"Processing {len(dataset)} windows across {d_sae} dimension(s)")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        drop_last=False,
        collate_fn=dim_collate,
    )

    generate_dim_data(
        dataloader,
        dec_b,
        enc_w,
        enc_b,
        threshold,
        n_windows_per_dim,
        original_dim_ids,
        device,
        output_store,
        seq_len,
        compute_dtype,
        show_progress=show_progress,
    )

    print(f"Completed! Data saved to {output_store}")
    return output_store

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate SAE latents for specified dimensions."
    )
    parser.add_argument(
        "--indices-file",
        required=True,
        help="Path to the .npy file containing top index positions per latent dimension.",
    )
    parser.add_argument(
        "--dim-range",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        help="Inclusive range of latent dimensions to process. Omit to process all.",
    )
    parser.add_argument(
        "--latents-file",
        help="Optional path to the .npy file containing latents. Currently unused but kept for compatibility.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the tqdm progress bar (prints periodic updates instead).",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.dim_range:
        start, end = args.dim_range
        if start > end:
            raise ValueError("START must be less than or equal to END for --dim-range.")
        dims = list(range(start, end + 1))
    else:
        dims = None

    setup_and_generate(
        dims=dims,
        latents_file=args.latents_file,
        indices_file=args.indices_file,
        show_progress=not args.no_progress_bar,
    )


if __name__ == '__main__':
    main()
