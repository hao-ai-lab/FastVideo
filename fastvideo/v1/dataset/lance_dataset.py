# lance_dataset.py
"""
PyTorch IterableDataset for loading preprocessed video data (VAE latents,
T5 embeddings, masks) from a Lance dataset.

This version iterates row-by-row within Lance batches, reconstructs tensors
individually, and yields single processed items. It does NOT include
Classifier-Free Guidance logic.

Assumes preprocessing produces a Lance file conforming to schema.py.
"""

import lance
import lance.torch.data
import torch
from torch.utils.data import DataLoader, IterableDataset
import numpy as np
import os
import time
import logging
from typing import Tuple, Optional, Iterator, List

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def reconstruct_tensor(
    bytes_val: Optional[bytes],
    shape_val: Optional[List[int]],
    dtype_str: Optional[str],
) -> Optional[torch.Tensor]:
    """Reconstructs a torch.Tensor from Lance data components."""
    if bytes_val is None or shape_val is None or dtype_str is None:
        return None
    try:
        dtype_map = {
            "float64": np.float64,
            "float32": np.float32,
            "float16": np.float16,
            "int64": np.int64,
            "int32": np.int32,
            "int16": np.int16,
            "int8": np.int8,
            "uint64": np.uint64,
            "uint32": np.uint32,
            "uint16": np.uint16,
            "uint8": np.uint8,
        }
        if dtype_str == "bool":
            np_arr = np.frombuffer(bytes_val, dtype=np.uint8).reshape(
                tuple(shape_val)
            )
            tensor = torch.from_numpy(np_arr.copy()).to(torch.bool)
        elif dtype_str == "bfloat16":
            # It's more efficient and lossless to save/load bfloat tensors
            # as uint16 given np does not natively support bfloat16
            uint16_arr = np.frombuffer(bytes_val, dtype=np.uint16).reshape(
                tuple(shape_val)
            )
            tensor = torch.from_numpy(uint16_arr.copy()).view(torch.bfloat16)
        elif dtype_str in dtype_map:
            np_dtype = dtype_map[dtype_str]
            np_arr = np.frombuffer(bytes_val, dtype=np_dtype).reshape(
                tuple(shape_val)
            )
            tensor = torch.from_numpy(np_arr.copy())
        else:
            logger.error(
                f"Unsupported dtype string for reconstruction: {dtype_str}"
            )
            return None
        return tensor
    except Exception as e:
        logger.error(
            f"Tensor reconstruction failed: {e}. Shape: {shape_val}, Dtype: {dtype_str}",
            exc_info=False,
        )
        return None


class LanceLatentDataset(IterableDataset):
    """
    Loads VAE latents and text embeddings/masks from a Lance dataset, yielding
    individual processed items. Applies VAE latent slicing.

    Args:
        lance_dataset_path (str): Path to the Lance dataset directory.
        num_latent_t (int): The number of temporal frames (T) to slice from VAE latents.
        lance_internal_batch_size (int): Batch size for Lance's internal reading efficiency.
        lance_num_workers (int): Number of worker processes for Lance reader.
        lance_batch_readahead (int): Number of batches for Lance reader to read ahead.
        filter (Optional[str]): Lance filter string to apply.
    """

    def __init__(
        self,
        lance_dataset_path: str,
        num_latent_t: int,
        lance_internal_batch_size: int = 64,
        lance_num_workers: int = 0,
        lance_batch_readahead: Optional[int] = None,
        filter: Optional[str] = None,
    ):
        super().__init__()
        if not os.path.exists(lance_dataset_path):
            raise FileNotFoundError(
                f"Lance path not found: {lance_dataset_path}"
            )

        self.lance_dataset_path = lance_dataset_path
        self.num_latent_t = num_latent_t
        self.lance_internal_batch_size = lance_internal_batch_size
        self.lance_num_workers = lance_num_workers
        self.filter = filter

        if lance_batch_readahead is None:
            self.lance_batch_readahead = (
                max(4, self.lance_num_workers * 2)
                if self.lance_num_workers > 0
                else 4
            )
        else:
            self.lance_batch_readahead = lance_batch_readahead

        # Define columns needed for reconstruction
        self.columns_to_load = [
            "id",
            "vae_latent_bytes",
            "vae_latent_shape",
            "vae_latent_dtype",
            "text_embedding_bytes",
            "text_embedding_shape",
            "text_embedding_dtype",
            "text_attention_mask_bytes",
            "text_attention_mask_shape",
            "text_attention_mask_dtype",
            "caption",  # for debugging
        ]

        # Basic dataset info
        try:
            self._dataset_info = lance.dataset(
                self.lance_dataset_path, mode="read"
            )
            self._dataset_len = len(self._dataset_info)
            # Verify schema contains required columns
            schema_fields = {field.name for field in self._dataset_info.schema}
            missing_cols = [
                col for col in self.columns_to_load if col not in schema_fields
            ]
            if missing_cols:
                raise ValueError(
                    f"Lance schema missing required columns: {missing_cols}"
                )
            logger.info(
                f"Opened Lance dataset: {self.lance_dataset_path} ({self._dataset_len} rows)"
            )
        except Exception as e:
            logger.error(f"Failed to open Lance dataset: {e}", exc_info=True)
            raise

    def _create_lance_iterator(self):
        """Creates the LanceDataset iterator."""
        try:
            # Note: batch_size here is for Lance's internal reads
            lance_ds_reader = lance.torch.data.LanceDataset(
                self.lance_dataset_path,
                columns=self.columns_to_load,
                batch_size=self.lance_internal_batch_size,
                filter=self.filter,
                batch_readahead=self.lance_batch_readahead,
                # num_workers=self.lance_num_workers, # Handled by Lance env vars usually
                to_tensor_fn=None,  # We process the pyarrow batch manually
            )
            return lance_ds_reader
        except Exception as e:
            logger.error(
                f"Failed to create LanceDataset reader instance: {e}",
                exc_info=True,
            )
            raise

    def __iter__(
        self,
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Iterates through Lance batches, processes rows individually, and yields items.
        """
        lance_iterator = self._create_lance_iterator()

        for record_batch in lance_iterator:
            # record_batch is pyarrow.RecordBatch
            try:
                rows_as_dicts = record_batch.to_pylist()
            except Exception as e:
                logger.warning(
                    f"Error converting RecordBatch to pylist: {e}",
                    exc_info=False,
                )
                continue

            for row_dict in rows_as_dicts:
                # --- Reconstruct tensors for this row ---
                latent = reconstruct_tensor(
                    row_dict.get("vae_latent_bytes"),
                    row_dict.get("vae_latent_shape"),
                    row_dict.get("vae_latent_dtype"),
                )
                prompt_embed = reconstruct_tensor(
                    row_dict.get("text_embedding_bytes"),
                    row_dict.get("text_embedding_shape"),
                    row_dict.get("text_embedding_dtype"),
                )
                prompt_attention_mask = reconstruct_tensor(
                    row_dict.get("text_attention_mask_bytes"),
                    row_dict.get("text_attention_mask_shape"),
                    row_dict.get("text_attention_mask_dtype"),
                )

                if (
                    latent is None
                    or prompt_embed is None
                    or prompt_attention_mask is None
                ):
                    item_id = row_dict.get("id", "Unknown")
                    logger.warning(
                        f"Skipping item id={item_id} due to reconstruction failure."
                    )
                    continue

                # --- Apply Transformations ---
                # 1. Slice latent temporal dimension (assuming shape [C, T, H, W])
                if latent.dim() == 4 and latent.shape[1] >= self.num_latent_t:
                    latent = latent[:, -self.num_latent_t :, :, :]
                elif latent.dim() == 4 and latent.shape[1] < self.num_latent_t:
                    logger.warning(
                        f"Item id={row_dict.get('id', 'Unknown')} has fewer latent frames ({latent.shape[1]}) than requested ({self.num_latent_t}). Skipping."
                    )
                    continue
                else:
                    logger.warning(
                        f"Unexpected latent dimension ({latent.dim()}). Skipping item id={row_dict.get('id', 'Unknown')}."
                    )
                    continue

                # --- Ensure mask is boolean ---
                prompt_attention_mask = prompt_attention_mask.bool()

                # --- Yield the processed item tuple ---
                yield latent, prompt_embed, prompt_attention_mask

    def __len__(self):
        """Return the total number of items in the dataset."""
        return self._dataset_len


def lance_latent_collate_function(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Collates a batch of individual (latent, prompt_embed, prompt_mask) tuples.
    Pads latents to the maximum size in the batch and creates latent attention masks.
    Stacks prompt embeddings and masks (assumed to be consistently shaped).
    """
    if not batch:
        return None

    try:
        latents, prompt_embeds, prompt_attention_masks = zip(*batch)
    except Exception as e:
        logger.error(f"Error unzipping batch in collate function: {e}")
        return None

    # --- Pad Latents and Create Latent Attention Mask ---
    if not latents or not all(isinstance(l, torch.Tensor) for l in latents):
        logger.error("Invalid or empty latents found during collation.")
        return None
    if not all(
        l.dim() == 4 for l in latents
    ):  # Expecting C, T, H, W after slicing
        logger.error(
            "Inconsistent latent dimensions found. Expected 4D (C,T,H,W)."
        )
        return None

    try:
        max_t = max(l.shape[1] for l in latents)
        max_h = max(l.shape[2] for l in latents)
        max_w = max(l.shape[3] for l in latents)
        original_shapes = [
            (l.shape[1], l.shape[2], l.shape[3]) for l in latents
        ]
    except IndexError:
        logger.error("Error accessing latent shapes during collation.")
        return None

    padded_latents_list = []
    for latent in latents:
        pad_t = max_t - latent.shape[1]
        pad_h = max_h - latent.shape[2]
        pad_w = max_w - latent.shape[3]
        if pad_t < 0 or pad_h < 0 or pad_w < 0:  # Should not happen
            logger.error("Negative padding calculated in collate function.")
            return None
        padded = torch.nn.functional.pad(
            latent, (0, pad_w, 0, pad_h, 0, pad_t), mode="constant", value=0
        )
        padded_latents_list.append(padded)

    try:
        latents_tensor = torch.stack(padded_latents_list, dim=0)
    except Exception as e:
        logger.error(f"Error stacking padded latents: {e}")
        return None

    latent_attn_mask = torch.zeros(
        len(latents),
        max_t,
        max_h,
        max_w,
        dtype=torch.bool,
        device=latents_tensor.device,
    )
    for i, (t, h, w) in enumerate(original_shapes):
        latent_attn_mask[i, :t, :h, :w] = True  # True where valid

    # --- Stack Prompt Embeddings and Masks ---
    try:
        # Assume prompts/masks have consistent shape (e.g., fixed sequence length)
        prompt_embeds_tensor = torch.stack(prompt_embeds, dim=0)
        # Ensure masks are boolean
        prompt_attention_masks_tensor = torch.stack(
            [m.bool() for m in prompt_attention_masks], dim=0
        )
    except RuntimeError as e:
        logger.error(
            f"Error stacking prompt embeddings/masks: {e}. Check sequence length consistency."
        )
        return None
    except Exception as e:
        logger.error(
            f"Unexpected error during prompt collation: {e}", exc_info=True
        )
        return None

    # Final check for boolean masks
    if prompt_attention_masks_tensor.dtype != torch.bool:
        prompt_attention_masks_tensor = prompt_attention_masks_tensor.bool()
    if latent_attn_mask.dtype != torch.bool:
        latent_attn_mask = latent_attn_mask.bool()

    return (
        latents_tensor,
        prompt_embeds_tensor,
        latent_attn_mask,
        prompt_attention_masks_tensor,
    )


if __name__ == "__main__":
    DUMMY_DATA_PATH = "./dummy_lance_video_data.lance"
    if not os.path.exists(DUMMY_DATA_PATH):
        print(f"Error: Dummy dataset not found at {DUMMY_DATA_PATH}")
        print(
            "Please install latest lance `pip install --pre --extra-index-url https://pypi.fury.io/lancedb/ pylance`"
        )
        print("and run 'python generate_dummy_data.py' first.")
        exit(1)

    NUM_LATENT_T_SLICE = 28
    DATALOADER_BATCH_SIZE = 8  # Final batch size for the model iterator
    LANCE_INTERNAL_BATCH_SIZE = 64  # Lance internal read size for efficiency
    LANCE_NUM_WORKERS = 0  # Set > 0 for Lance multiprocessing reading
    LANCE_SHUFFLE = True
    EPOCHS_TO_RUN = 1

    # --- Create Dataset instance ---
    try:
        dataset = LanceLatentDataset(
            lance_dataset_path=DUMMY_DATA_PATH,
            num_latent_t=NUM_LATENT_T_SLICE,
            lance_internal_batch_size=LANCE_INTERNAL_BATCH_SIZE,
            lance_num_workers=LANCE_NUM_WORKERS,
            shuffle=LANCE_SHUFFLE,
            # seed=42 # Optional
        )
        logger.info(f"Dataset length: {len(dataset)}")
    except Exception as e:
        logger.error(f"Failed to create dataset: {e}", exc_info=True)
        exit(1)

    # --- Create DataLoader ---
    # DataLoader batch_size groups items yielded by dataset's __iter__
    pytorch_dataloader_workers = 0 if LANCE_NUM_WORKERS > 0 else 2
    if LANCE_NUM_WORKERS > 0 and pytorch_dataloader_workers > 0:
        logger.warning(
            "Set PyTorch DataLoader num_workers=0 when Lance workers > 0."
        )
        pytorch_dataloader_workers = 0

    dataloader = DataLoader(
        dataset,
        batch_size=DATALOADER_BATCH_SIZE,  # Group individual items
        num_workers=pytorch_dataloader_workers,
        collate_fn=lance_latent_collate_function,  # Use the padding collate fn
        pin_memory=torch.cuda.is_available(),
    )

    # --- Iterate through DataLoader ---
    logger.info(f"Starting iteration for {EPOCHS_TO_RUN} epochs...")
    total_batches = 0
    total_items = 0
    start_time = time.time()

    for epoch in range(EPOCHS_TO_RUN):
        logger.info(f"--- Epoch {epoch} ---")
        items_in_epoch = 0
        for batch_idx, batch_data in enumerate(dataloader):
            if batch_data is None:
                logger.warning(
                    f"Skipping invalid batch {batch_idx} in epoch {epoch}."
                )
                continue

            latents, prompt_embeds, latent_attn_mask, prompt_attention_masks = (
                batch_data
            )
            current_batch_size = latents.shape[0]
            total_batches += 1
            items_in_epoch += current_batch_size

            if batch_idx == 0 and epoch == 0:  # Print first batch info
                print(f"  First Batch Shapes/Types:")
                print(
                    f"  Latents Shape          : {latents.shape} ({latents.dtype})"
                )
                print(
                    f"  Prompt Embeds Shape    : {prompt_embeds.shape} ({prompt_embeds.dtype})"
                )
                print(
                    f"  Latent Mask Shape      : {latent_attn_mask.shape} ({latent_attn_mask.dtype})"
                )
                print(
                    f"  Prompt Mask Shape      : {prompt_attention_masks.shape} ({prompt_attention_masks.dtype})"
                )

            time.sleep(0.01)  # Simulate work

            if (batch_idx + 1) % 20 == 0:
                logger.info(
                    f"  Epoch {epoch}, Batch {batch_idx+1}, Items Processed (Epoch): {items_in_epoch}"
                )

        total_items += items_in_epoch
        logger.info(f"--- Epoch {epoch} Summary ---")
        logger.info(f"  Batches Processed: {batch_idx + 1}")
        logger.info(f"  Items Processed: {items_in_epoch}")

    end_time = time.time()
    duration = end_time - start_time
    avg_items_per_sec = total_items / duration if duration > 0 else 0

    logger.info("\n--- Iteration Finished ---")
    logger.info(f"Total Epochs Run: {EPOCHS_TO_RUN}")
    logger.info(f"Total Batches Processed: {total_batches}")
    logger.info(f"Total Items Processed: {total_items}")
    logger.info(f"Total Duration: {duration:.2f} seconds")
    logger.info(f"Average Items/sec: {avg_items_per_sec:.2f}")
