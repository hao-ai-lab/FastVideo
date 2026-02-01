# SPDX-License-Identifier: Apache-2.0
# HYWorld Camera Dataset with Memory Training Support
# Adapted from HY-WorldPlay trainer

import json
import os
import random
from typing import Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader

from fastvideo.distributed import (
    get_local_torch_device,
    get_sp_world_size,
    get_world_rank,
    get_world_size,
)
from fastvideo.logger import init_logger
from fastvideo.models.dits.hyworld.pose import (
    camera_center_normalization,
    one_hot_to_one_dimension,
)
from fastvideo.models.dits.hyworld.retrieval_context import (
    generate_points_in_sphere,
    select_aligned_memory_frames,
)

logger = init_logger(__name__)


class HYWorldBatchSampler(Sampler[list[int]]):
    """
    A batch sampler that handles distributed training with sequence parallelism.
    """

    def __init__(
        self,
        batch_size: int,
        dataset_size: int,
        num_sp_groups: int,
        sp_world_size: int,
        global_rank: int,
        drop_last: bool = True,
        drop_first_row: bool = False,
        seed: int = 0,
    ):
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.drop_last = drop_last
        self.seed = seed
        self.num_sp_groups = num_sp_groups
        self.global_rank = global_rank
        self.sp_world_size = sp_world_size

        rng = torch.Generator().manual_seed(self.seed)
        global_indices = torch.randperm(self.dataset_size, generator=rng)

        if drop_first_row:
            global_indices = global_indices[global_indices != 0]
            self.dataset_size = self.dataset_size - 1

        if self.drop_last:
            num_batches = self.dataset_size // self.batch_size
            num_global_batches = num_batches // self.num_sp_groups
            global_indices = global_indices[:num_global_batches *
                                             self.num_sp_groups *
                                             self.batch_size]
        else:
            if self.dataset_size % (self.num_sp_groups * self.batch_size) != 0:
                padding_size = self.num_sp_groups * self.batch_size - (
                    self.dataset_size % (self.num_sp_groups * self.batch_size))
                logger.info("Padding dataset from %d to %d",
                           self.dataset_size, self.dataset_size + padding_size)
                global_indices = torch.cat(
                    [global_indices, global_indices[:padding_size]])

        ith_sp_group = self.global_rank // self.sp_world_size
        sp_group_local_indices = global_indices[ith_sp_group::self.num_sp_groups]
        self.sp_group_local_indices = sp_group_local_indices
        logger.info("Dataset size for each sp group: %d",
                   len(sp_group_local_indices))

    def __iter__(self):
        indices = self.sp_group_local_indices
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield batch_indices.tolist()

    def __len__(self):
        return len(self.sp_group_local_indices) // self.batch_size


class HYWorldCameraDataset(Dataset):
    """
    Dataset for HYWorld training with camera pose and memory support.
    """

    def __init__(
        self,
        json_path: str,
        causal: bool,
        window_frames: int,
        batch_size: int,
        cfg_rate: float,
        i2v_rate: float,
        drop_last: bool,
        drop_first_row: bool,
        seed: int,
        device,
        shared_state,
        neg_prompt_path: str = None,
        neg_byt5_path: str = None,
    ):
        self.json_data = json.load(open(json_path, 'r'))
        self.all_length = len(self.json_data)
        self.causal = causal
        self.window_frames = window_frames
        self.memory_frames = 20
        self.cfg_rate = cfg_rate
        self.rng = random.Random(seed)
        self.i2v_rate = i2v_rate
        self.device = device
        self.shared_state = shared_state

        self.sampler = HYWorldBatchSampler(
            batch_size=batch_size,
            dataset_size=self.all_length,
            num_sp_groups=get_world_size() // get_sp_world_size(),
            sp_world_size=get_sp_world_size(),
            global_rank=get_world_rank(),
            drop_last=drop_last,
            drop_first_row=drop_first_row,
            seed=seed,
        )

        self.points_local = generate_points_in_sphere(50000, 8.0).to(device)

        # Load negative prompts if provided
        self.neg_prompt_pt = None
        self.neg_byt5_pt = None
        if neg_prompt_path and os.path.exists(neg_prompt_path):
            self.neg_prompt_pt = torch.load(neg_prompt_path, map_location="cpu", weights_only=True)
        if neg_byt5_path and os.path.exists(neg_byt5_path):
            self.neg_byt5_pt = torch.load(neg_byt5_path, map_location="cpu", weights_only=True)

    def __len__(self):
        return self.all_length

    def update_max_frames(self, training_step):
        """Progressive training: increase max frames as training progresses."""
        if training_step < 500:
            self.shared_state["max_frames"] = 32
        elif training_step < 1000:
            self.shared_state["max_frames"] = 64
        elif training_step < 2000:
            self.shared_state["max_frames"] = 96
        elif training_step < 3000:
            self.shared_state["max_frames"] = 128
        else:
            self.shared_state["max_frames"] = 160

    def __getitem__(self, idx):
        while True:
            # try:
            if True:
                json_data = self.json_data[idx]
                latent_pt_path = json_data['latent_path']
                pose_path = json_data['pose_path']

                latent_pt = torch.load(
                    os.path.join(latent_pt_path),
                    map_location="cpu",
                    weights_only=True,
                )
                latent = latent_pt['latent'][0]
                latent_length = latent.shape[1]

                if latent_length < self.window_frames:
                    idx = self.rng.randint(0, self.all_length - 1)
                    continue
                else:
                    max_frames = int(self.shared_state["max_frames"]) // 4 * 4
                    max_length = min(max_frames, latent_length // 4 * 4)

                latent = latent[:, :max_length, ...]

                prompt_embed = latent_pt['prompt_embeds'][0]
                prompt_mask = latent_pt['prompt_mask'][0]

                image_cond = latent_pt['image_cond'][0]
                vision_states = latent_pt['vision_states'][0]
                byt5_text_states = latent_pt['byt5_text_states'][0]
                byt5_text_mask = latent_pt['byt5_text_mask'][0]

                # Apply CFG (classifier-free guidance) with probability cfg_rate
                if self.rng.random() < self.cfg_rate and self.neg_prompt_pt is not None:
                    prompt_embed = self.neg_prompt_pt['negative_prompt_embeds'][0]
                    prompt_mask = self.neg_prompt_pt['negative_prompt_mask'][0]
                    if self.neg_byt5_pt is not None:
                        byt5_text_states = self.neg_byt5_pt['byt5_text_states'][0]
                        byt5_text_mask = self.neg_byt5_pt['byt5_text_mask'][0]

                # Load pose data
                pose_json = json.load(open(pose_path, 'r'))
                pose_keys = list(pose_json.keys())
                intrinsic_list = []
                w2c_list = []
                for i in range(latent.shape[1]):
                    t_key = pose_keys[0] if i == 0 else pose_keys[4 * (i - 1) + 4]
                    intrinsic = np.array(pose_json[t_key]['intrinsic'])
                    w2c = np.array(pose_json[t_key]['w2c'])

                    intrinsic[0, 0] /= intrinsic[0, 2] * 2
                    intrinsic[1, 1] /= intrinsic[1, 2] * 2
                    intrinsic[0, 2] = 0.5
                    intrinsic[1, 2] = 0.5
                    w2c_list.append(w2c)
                    intrinsic_list.append(intrinsic)

                w2c_list = np.array(w2c_list)
                w2c_list = camera_center_normalization(w2c_list)
                intrinsic_list = torch.tensor(np.array(intrinsic_list))

                # Compute action labels
                if 'action_path' in json_data and os.path.exists(json_data["action_path"]):
                    # Load pre-computed action labels
                    trans_one_hot = np.zeros((intrinsic_list.shape[0], 4), dtype=np.int32)
                    rotate_one_hot = np.zeros((intrinsic_list.shape[0], 4), dtype=np.int32)
                    action_json = json.load(open(json_data["action_path"], 'r'))
                    action_keys = list(action_json.keys())
                    for action_idx in range(1, trans_one_hot.shape[0]):
                        t_key = action_keys[4 * (action_idx - 1) + 4]
                        t_move_action = action_json[t_key]["move_action"]
                        t_view_action = action_json[t_key]["view_action"]
                        if "W" in t_move_action and "S" not in t_move_action:
                            trans_one_hot[action_idx, 0] = 1
                        if "S" in t_move_action and "W" not in t_move_action:
                            trans_one_hot[action_idx, 1] = 1
                        if "D" in t_move_action and "A" not in t_move_action:
                            trans_one_hot[action_idx, 2] = 1
                        if "A" in t_move_action and "D" not in t_move_action:
                            trans_one_hot[action_idx, 3] = 1

                        if t_view_action == "LR":
                            rotate_one_hot[action_idx, 0] = 1
                        elif t_view_action == "LL":
                            rotate_one_hot[action_idx, 1] = 1
                        elif t_view_action == "LU":
                            rotate_one_hot[action_idx, 2] = 1
                        elif t_view_action == "LD":
                            rotate_one_hot[action_idx, 3] = 1

                    trans_one_label = one_hot_to_one_dimension(torch.tensor(trans_one_hot))
                    rotate_one_label = one_hot_to_one_dimension(torch.tensor(rotate_one_hot))
                    action_for_pe = trans_one_label * 9 + rotate_one_label
                else:
                    # Compute action labels from camera poses on the fly
                    c2ws = np.linalg.inv(w2c_list)
                    C_inv = np.linalg.inv(c2ws[:-1])
                    relative_c2w = np.zeros_like(c2ws)
                    relative_c2w[0, ...] = c2ws[0, ...]
                    relative_c2w[1:, ...] = C_inv @ c2ws[1:, ...]
                    trans_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)
                    rotate_one_hot = np.zeros((relative_c2w.shape[0], 4), dtype=np.int32)

                    move_norm_valid = 0.01
                    for i in range(1, relative_c2w.shape[0]):
                        move_dirs = relative_c2w[i, :3, 3]
                        move_norms = np.linalg.norm(move_dirs)
                        if move_norms > move_norm_valid:
                            move_norm_dirs = move_dirs / move_norms
                            angles_rad = np.arccos(move_norm_dirs.clip(-1.0, 1.0))
                            trans_angles_deg = angles_rad * (180.0 / torch.pi)

                            if trans_angles_deg[2] < 60:
                                trans_one_hot[i, 0] = 1
                            elif trans_angles_deg[2] > 120:
                                trans_one_hot[i, 1] = 1

                            if trans_angles_deg[0] < 60:
                                trans_one_hot[i, 2] = 1
                            elif trans_angles_deg[0] > 120:
                                trans_one_hot[i, 3] = 1

                        R_rel = relative_c2w[i, :3, :3]
                        r = R.from_matrix(R_rel)
                        rot_angles_deg = r.as_euler('xyz', degrees=True)

                        if rot_angles_deg[1] > 5e-2:
                            rotate_one_hot[i, 0] = 1
                        elif rot_angles_deg[1] < -5e-2:
                            rotate_one_hot[i, 1] = 1

                        if rot_angles_deg[0] > 5e-2:
                            rotate_one_hot[i, 2] = 1
                        elif rot_angles_deg[0] < -5e-2:
                            rotate_one_hot[i, 3] = 1

                    trans_one_hot = torch.tensor(trans_one_hot)
                    rotate_one_hot = torch.tensor(rotate_one_hot)

                    trans_one_label = one_hot_to_one_dimension(trans_one_hot)
                    rotate_one_label = one_hot_to_one_dimension(rotate_one_hot)
                    action_for_pe = trans_one_label * 9 + rotate_one_label

                # Memory training: select frames outside window with probability
                select_window_out_flag = 0
                select_prob = self.rng.random()
                selected_history_frame_id = None
                current_frame_idx = None
                temporal_context_size = 12

                if select_prob < 0.8:
                    select_window_out_flag = 1  # Select frames outside window
                    # max_index = latent.shape[1] - (self.window_frames - self.memory_frames)

                    # start_chunk_id = (self.window_frames) // 4
                    # end_chunk_id = max_index // 4
                    # current_frame_idx = self.rng.randint(start_chunk_id, end_chunk_id) * 4
                    # IMPORTANT:
                    # `w2c_list` / `intrinsic_list` are built at *latent* resolution:
                    #   len(w2c_list) == latent.shape[1]  (e.g. 32 for 125-frame videos)
                    # Therefore `current_frame_idx` must be a valid latent index in [0, latent_T).
                    #
                    # The original implementation computed `max_index` using `(window_frames - memory_frames)`.
                    # When `memory_frames > window_frames` (e.g. memory_frames=20, window_frames=16),
                    # `max_index` can exceed `latent_T`, and with `randint` (inclusive upper bound)
                    # this can yield out-of-range indices like 32/36, triggering:
                    #   "current frame index ... {current_frame_idx}, {len(w2c_list)}"
                    #
                    # We sample a chunk-aligned latent start index safely:
                    #   current_frame_idx in {window_frames, window_frames+4, ..., latent_T-4}
                    pred_latent_size = 4
                    latent_T = latent.shape[1]
                    start_idx = (self.window_frames // pred_latent_size) * pred_latent_size
                    max_start = latent_T - pred_latent_size
                    max_start = (max_start // pred_latent_size) * pred_latent_size
                    if max_start < start_idx:
                        # Not enough latents to select an "outside-window" chunk; fall back to in-window.
                        select_window_out_flag = 0
                    else:
                        current_frame_idx = self.rng.randrange(
                            start_idx, max_start + pred_latent_size, pred_latent_size
                        )

                    if select_window_out_flag == 1:
                        selected_history_frame_id = select_aligned_memory_frames(
                            w2c_list,
                            current_frame_idx,
                            memory_frames=self.memory_frames,
                            temporal_context_size=temporal_context_size,
                            pred_latent_size=4,
                            points_local=self.points_local,
                            device=self.device
                        )
                        selected_history_frame_id.extend(range(current_frame_idx, current_frame_idx + 4))
                        latent = latent[:, selected_history_frame_id]
                        w2c_list = w2c_list[selected_history_frame_id]
                        intrinsic_list = intrinsic_list[selected_history_frame_id]
                        action_for_pe = action_for_pe[selected_history_frame_id]
                else:
                    pred_latent_size = self.window_frames
                    latent = latent[:, :pred_latent_size, ...]
                    w2c_list = w2c_list[:pred_latent_size]
                    intrinsic_list = intrinsic_list[:pred_latent_size]
                    action_for_pe = action_for_pe[:pred_latent_size]

                i2v_mask = torch.ones_like(latent)

                batch = {
                    "i2v_mask": i2v_mask,
                    "latent": latent,
                    "prompt_embed": prompt_embed,
                    "w2c": torch.tensor(w2c_list),
                    "intrinsic": intrinsic_list,
                    "action": action_for_pe,
                    "action_for_pe": action_for_pe,
                    "context_frames_list": None,
                    "select_window_out_flag": select_window_out_flag,
                    # Debug/visualization metadata (latent indices in the ORIGINAL sequence before repacking).
                    # - in-window: both are None
                    # - out-window: `selected_history_frame_id` includes history indices AND the current chunk indices
                    #              appended at the end (range(current_frame_idx, current_frame_idx+4)).
                    "selected_history_frame_id": selected_history_frame_id,
                    "current_frame_idx": current_frame_idx,
                    "temporal_context_size": temporal_context_size,
                    "video_path": json_data["pose_path"],
                    "max_length": max_frames,
                    "image_cond": image_cond,
                    "vision_states": vision_states,
                    "prompt_mask": prompt_mask,
                    "byt5_text_states": byt5_text_states,
                    "byt5_text_mask": byt5_text_mask,
                }
                break
            # except Exception as e:
            #     logger.warning(f'Error loading sample {idx}: {e}')
            #     idx = self.rng.randint(0, self.all_length - 1)
        return batch


def hyworld_collate_fn(batch):
    """Collate function for HYWorld dataset."""
    latent = torch.stack([b["latent"] for b in batch], dim=0)
    prompt_embed = torch.stack([b["prompt_embed"] for b in batch], dim=0)
    w2c = torch.stack([b["w2c"] for b in batch], dim=0)
    intrinsic = torch.stack([b["intrinsic"] for b in batch], dim=0)
    action = torch.stack([b["action"] for b in batch], dim=0)
    action_for_pe = torch.stack([b["action_for_pe"] for b in batch], dim=0)
    i2v_mask = torch.stack([b["i2v_mask"] for b in batch], dim=0)

    image_cond = torch.stack([b["image_cond"] for b in batch], dim=0)
    vision_states = torch.stack([b["vision_states"] for b in batch], dim=0)
    prompt_mask = torch.stack([b["prompt_mask"] for b in batch], dim=0)
    byt5_text_states = torch.stack([b["byt5_text_states"] for b in batch], dim=0)
    byt5_text_mask = torch.stack([b["byt5_text_mask"] for b in batch], dim=0)

    context_frames_list = [b["context_frames_list"] for b in batch]
    select_window_out_flag = [b["select_window_out_flag"] for b in batch]
    selected_history_frame_id = [b.get("selected_history_frame_id") for b in batch]
    current_frame_idx = [b.get("current_frame_idx") for b in batch]
    temporal_context_size = [b.get("temporal_context_size") for b in batch]
    video_path = [b["video_path"] for b in batch]
    max_length = [b["max_length"] for b in batch]

    return {
        "i2v_mask": i2v_mask,
        "latent": latent,
        "prompt_embed": prompt_embed,
        "w2c": w2c,
        "intrinsic": intrinsic,
        "action": action,
        "video_path": video_path,
        "context_frames_list": context_frames_list,
        "select_window_out_flag": select_window_out_flag,
        "selected_history_frame_id": selected_history_frame_id,
        "current_frame_idx": current_frame_idx,
        "temporal_context_size": temporal_context_size,
        "action_for_pe": action_for_pe,
        "max_length": max_length,
        "image_cond": image_cond,
        "vision_states": vision_states,
        "prompt_mask": prompt_mask,
        "byt5_text_states": byt5_text_states,
        "byt5_text_mask": byt5_text_mask,
    }


def build_hyworld_camera_dataloader(
    json_path: str,
    causal: bool,
    window_frames: int,
    batch_size: int,
    num_data_workers: int,
    drop_last: bool,
    drop_first_row: bool,
    seed: int,
    cfg_rate: float,
    i2v_rate: float,
    neg_prompt_path: str = None,
    neg_byt5_path: str = None,
) -> Tuple[HYWorldCameraDataset, StatefulDataLoader]:
    """
    Build dataloader for HYWorld camera dataset.

    Args:
        json_path: Path to JSON file containing dataset info.
        causal: Whether to use causal attention.
        window_frames: Window size for training.
        batch_size: Batch size.
        num_data_workers: Number of data loading workers.
        drop_last: Whether to drop the last incomplete batch.
        drop_first_row: Whether to drop the first row.
        seed: Random seed.
        cfg_rate: Classifier-free guidance rate.
        i2v_rate: Image-to-video rate.
        neg_prompt_path: Path to negative prompt embeddings.
        neg_byt5_path: Path to negative ByT5 embeddings.

    Returns:
        Tuple of (dataset, dataloader).
    """
    manager = mp.Manager()
    shared_state = manager.dict()
    shared_state["max_frames"] = window_frames

    dataset = HYWorldCameraDataset(
        json_path=json_path,
        causal=causal,
        window_frames=window_frames,
        batch_size=batch_size,
        cfg_rate=cfg_rate,
        i2v_rate=i2v_rate,
        drop_last=drop_last,
        drop_first_row=drop_first_row,
        seed=seed,
        device=get_local_torch_device(),
        shared_state=shared_state,
        neg_prompt_path=neg_prompt_path,
        neg_byt5_path=neg_byt5_path,
    )

    loader = StatefulDataLoader(
        dataset,
        batch_sampler=dataset.sampler,
        collate_fn=hyworld_collate_fn,
        num_workers=num_data_workers,
        pin_memory=True,
        persistent_workers=num_data_workers > 0,
    )
    return dataset, loader
