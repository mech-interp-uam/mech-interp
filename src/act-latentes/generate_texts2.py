import argparse
import uuid
import os
from acts import get_ds
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
import concurrent.futures
import queue
import json

# Stumbling blocks:
# 1. We attempted to store everything in json and in the main thread
# 2. Output switched from JSON to Parquet
# 3. Just like in generate_windows2.py, we started sorting the windows to
#    avoid running out of ram
# 4. We introduced ParquetWriter for faster frequent writes.

WINDOW_LEN = 64


class TokenWindowDataset(Dataset):
    def __init__(
        self,
        # sorted_indices, _ = indices.flatten().sort()
        sorted_indices: torch.Tensor,  # [total_windows]
        original_ds,
        window_len: int = WINDOW_LEN,
    ):
        self.sorted_indices = sorted_indices
        self.original_ds    = original_ds
        self.window_len     = window_len

    def __len__(self):
        return len(self.sorted_indices)

    def __getitem__(self, idx: int):
        """
        Get a single window of token IDs.
        Returns data in sorted order (by data index).
        Returns: (idx, ids_window) where idx is the position in sorted order
        """
        # Get the sorted data index
        data_idx = int(self.sorted_indices[idx].item())

        # Calculate window boundaries with FIXED window size
        start_idx = data_idx - self.window_len // 2
        end_idx   = start_idx + self.window_len

        if start_idx < 0:
            # Too close to start: shift window right
            start_idx = 0
            end_idx   = self.window_len
        elif end_idx > len(self.original_ds):
            # Too close to end: shift window left
            end_idx   = len(self.original_ds)
            start_idx = end_idx - self.window_len

        ids_window = self.original_ds[start_idx:end_idx]["token_id"]

        return (idx, ids_window)


def collate_with_tokenizer_closure(tokenizer):
    def collate(
        batch: list[tuple[int, np.ndarray]],
    ):
        """
        Collate function for sorted windows.
        batch: list of (sorted_idx, token_ids_window)

        All windows are guaranteed to be exactly window_len tokens (no padding needed).
        """
        # Unzip batch into separate lists
        sorted_indices_list, ids_windows_list = zip(*batch)

        # Convert to flat indices tensor
        sorted_indices = torch.tensor(sorted_indices_list, dtype=torch.int64)

        # Decode all windows
        texts = tokenizer.batch_decode(ids_windows_list)

        # Retokenize to get offset mapping
        tokenizer_dict = tokenizer(texts, return_offsets_mapping=True, add_special_tokens=False)
        offsets = tokenizer_dict['offset_mapping']

        return sorted_indices, texts, offsets

    return collate


def writer_worker(write_queue, schema, output_file):
    """
    Background writer thread that receives batches and writes them incrementally to parquet.
    Uses ParquetWriter to append directly to the output file, overlapping I/O with compute.
    """
    WRITE_BATCH_SIZE = 100  # Write every 100 windows
    accumulated_texts = []
    accumulated_offsets = []
    current_offset = 0  # Track sorted_idx offset for this batch

    def _prepare_offsets(offset_window):
        return [{"start": int(start), "end": int(end)} for start, end in offset_window]

    with pq.ParquetWriter(output_file, schema, compression="zstd", use_dictionary=True) as writer:
        while True:
            item = write_queue.get()

            if item is None:  # Sentinel - write remaining and stop
                if accumulated_texts:
                    # Write final partial batch
                    rows = []
                    for i, (text, offset) in enumerate(zip(accumulated_texts, accumulated_offsets)):
                        rows.append({
                            "sorted_idx": current_offset + i,
                            "text": text,
                            "offsets": _prepare_offsets(offset),
                        })

                    table = pa.Table.from_pylist(rows, schema=schema)
                    writer.write_table(table)

                write_queue.task_done()
                return  # Writer context manager will close the file

            # Accumulate batch
            texts, offsets = item
            accumulated_texts.extend(texts)
            accumulated_offsets.extend(offsets)

            # Write when batch is large enough
            if len(accumulated_texts) >= WRITE_BATCH_SIZE:
                rows = []
                for i, (text, offset) in enumerate(zip(accumulated_texts, accumulated_offsets)):
                    rows.append({
                        "sorted_idx": current_offset + i,
                        "text": text,
                        "offsets": _prepare_offsets(offset),
                    })

                table = pa.Table.from_pylist(rows, schema=schema)
                writer.write_table(table)

                current_offset += len(accumulated_texts)
                accumulated_texts = []
                accumulated_offsets = []

            write_queue.task_done()


def generate_dim_data(
    dataloader,
    output_file: str,
    perm: torch.Tensor,
    n_windows_per_dim: int,
    d_sae: int,
    original_dim_ids: list[int],
    show_progress: bool = True,
):
    """
    Process windows in sorted order and write to parquet in sorted order with periodic writes.
    Uses background thread with ParquetWriter to overlap I/O with compute.
    Metadata saved to sidecar JSON file for crash resilience.
    Use scripts/unsort_texts.py to convert to dimension-grouped format.
    """
    total_windows = len(dataloader.dataset)

    # Write metadata to sidecar file IMMEDIATELY (before processing starts)
    # This is critical for crash resilience - if the run crashes, metadata is already saved
    metadata_file = f"{output_file}.meta.json"
    metadata = {
        'perm': perm.tolist(),
        'n_windows_per_dim': n_windows_per_dim,
        'd_sae': d_sae,
        'original_dim_ids': original_dim_ids,
    }
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to {metadata_file}")

    # Schema for parquet (no metadata embedded)
    schema = pa.schema([
        ("sorted_idx", pa.int64()),
        ("text", pa.string()),
        ("offsets", pa.list_(pa.struct([
            ("start", pa.int32()),
            ("end", pa.int32()),
        ]))),
    ])

    # Start writer thread
    write_queue = queue.Queue(maxsize=50)  # Bounded queue to limit memory
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    writer_future = executor.submit(writer_worker, write_queue, schema, output_file)

    progress_bar = tqdm(total=total_windows, desc="Windows", leave=False) if show_progress else None
    log_interval = max(1, total_windows // 10) if total_windows else 1
    processed_windows = 0

    try:
        # Process all windows in sorted order
        for sorted_indices, texts, offsets in dataloader:
            batch_size = len(sorted_indices)

            # Send to writer thread (batches arrive in order, so no indexing needed)
            write_queue.put((texts, offsets))

            processed_windows += batch_size
            if progress_bar:
                progress_bar.update(batch_size)
            else:
                if processed_windows == batch_size or processed_windows % log_interval == 0 or processed_windows == total_windows:
                    print(f"Processed {processed_windows}/{total_windows} windows")

        # Send sentinel to stop writer
        write_queue.put(None)

        # Wait for writer to finish
        writer_future.result()

        if progress_bar:
            progress_bar.close()

        print(f"\nFinished writing to {output_file}")
        print(f"Metadata: {metadata_file}")
        print(f"To convert to dimension-grouped format, run:")
        print(f"  python scripts/unsort_texts.py --input {output_file}")

    finally:
        executor.shutdown(wait=True)


def setup_and_generate(
        indices_file: str,
        output_file: str | None = None,
        dims: list[int] = None,
        show_progress: bool = True,
    ):
    batch_size = 16  # Increased for better throughput
    num_workers = 12
    prefetch_factor = 4

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    ds = get_ds()
    ds.set_format('numpy')

    top_indices = np.load(indices_file)
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
    # Flatten: (n_windows, d_sae) -> (n_windows * d_sae,)
    indices_flat = indices_tensor.T.flatten()  # Transpose then flatten
    sorted_indices, perm = indices_flat.sort()
    # perm[i] tells us: sorted[i] came from original[perm[i]]
    # For unsorting: unsorted[perm[i]] = sorted[i], so we use perm directly

    # Generate output filename if not provided
    if output_file is None:
        # Create output directory if it doesn't exist
        output_dir = "generated_texts"
        os.makedirs(output_dir, exist_ok=True)

        # Build descriptive filename
        run_id = uuid.uuid4().hex[:8]
        if dims is not None:
            dim_info = f"dims{dims[0]}-{dims[-1]}"
            dim_count = len(dims)
        else:
            dim_info = f"dims0-{d_sae-1}"
            dim_count = d_sae

        # Extract input filename stem
        indices_basename = os.path.splitext(os.path.basename(indices_file))[0]
        input_id = indices_basename.replace("top_indices_", "").replace("top_", "")

        filename = f"texts_from_{input_id}_d{dim_count}_nwin{n_windows_per_dim}_{dim_info}_{run_id}.parquet"
        output_file = os.path.join(output_dir, filename)

    print(f"Output file: {output_file}")

    dataset = TokenWindowDataset(
        sorted_indices,
        ds,
        window_len=WINDOW_LEN,
    )

    print(f"Processing {len(dataset)} windows across {d_sae} dimension(s)")

    dataloader = DataLoader(
        dataset,
        batch_size      = batch_size,
        shuffle         = False,
        num_workers     = num_workers,
        prefetch_factor = prefetch_factor,
        pin_memory      = False,
        drop_last       = False,
        collate_fn      = collate_with_tokenizer_closure(tokenizer)
    )

    generate_dim_data(
        dataloader,
        output_file,
        perm,
        n_windows_per_dim,
        d_sae,
        original_dim_ids,
        show_progress=show_progress,
    )

    metadata_file = f"{output_file}.meta.json"
    print(f"Completed! Data saved to {output_file}")

    return {
        "output_file": output_file,
        "metadata_file": metadata_file
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate detokenized windows for specified latent dimensions."
    )
    parser.add_argument(
        "--indices-file",
        required = True,
        help     = "Path to the .npy file containing top index positions per latent dimension.",
    )
    parser.add_argument(
        "--output-file",
        default = None,
        help    = "Path to the output Parquet file. Defaults to texts_from_<input>_<dims>_<uuid>.parquet.",
    )
    parser.add_argument(
        "--dim-range",
        type    = int,
        nargs   = 2,
        metavar = ("START", "END"),
        help    = "Inclusive range of latent dimensions to process. Omit to process all.",
    )
    parser.add_argument(
        "--no-progress-bar",
        action = "store_true",
        help   = "Disable the tqdm progress bar (prints periodic updates instead).",
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

    output_file = args.output_file

    setup_and_generate(
        indices_file  = args.indices_file,
        output_file   = output_file,
        dims          = dims,
        show_progress = not args.no_progress_bar,
    )


if __name__ == "__main__":
    main()
