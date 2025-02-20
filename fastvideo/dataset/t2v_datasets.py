import json
import math
import os
import random
from collections import Counter
from os.path import join as opj

import numpy as np
import torch
import torchvision
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset

from fastvideo.utils.dataset_utils import DecordInit
from fastvideo.utils.logging_ import main_print


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class DataSetProg(metaclass=SingletonMeta):

    def __init__(self):
        self.cap_list = []
        self.elements = []
        self.num_workers = 1
        self.n_elements = 0
        self.worker_elements = dict()
        self.n_used_elements = dict()

    def set_cap_list(self, num_workers, cap_list, n_elements):
        self.num_workers = num_workers
        self.cap_list = cap_list
        self.n_elements = n_elements
        self.elements = list(range(n_elements))
        random.shuffle(self.elements)
        print(f"n_elements: {len(self.elements)}", flush=True)

        for i in range(self.num_workers):
            self.n_used_elements[i] = 0
            per_worker = int(math.ceil(len(self.elements) / float(self.num_workers)))
            start = i * per_worker
            end = min(start + per_worker, len(self.elements))
            self.worker_elements[i] = self.elements[start:end]

    def get_item(self, work_info):
        if work_info is None:
            worker_id = 0
        else:
            worker_id = work_info.id

        idx = self.worker_elements[worker_id][self.n_used_elements[worker_id] % len(self.worker_elements[worker_id])]
        self.n_used_elements[worker_id] += 1
        return idx


dataset_prog = DataSetProg()


def filter_resolution(h, w, max_h_div_w_ratio=17 / 16, min_h_div_w_ratio=8 / 16):
    if h / w <= max_h_div_w_ratio and h / w >= min_h_div_w_ratio:
        return True
    return False


class T2V_dataset(Dataset):

    def __init__(self, args, transform, temporal_sample, tokenizer, transform_topcrop):
        self.data = args.data_merge_path
        self.num_frames = args.num_frames
        self.train_fps = args.train_fps
        self.use_image_num = args.use_image_num
        self.transform = transform
        self.transform_topcrop = transform_topcrop
        self.temporal_sample = temporal_sample
        self.tokenizer = tokenizer
        self.text_max_length = args.text_max_length
        self.cfg = args.cfg
        self.speed_factor = args.speed_factor
        self.max_height = args.max_height
        self.max_width = args.max_width
        self.drop_short_ratio = args.drop_short_ratio
        assert self.speed_factor >= 1
        self.v_decoder = DecordInit()
        self.video_length_tolerance_range = args.video_length_tolerance_range
        self.support_Chinese = True
        if "mt5" not in args.text_encoder_name:
            self.support_Chinese = False

        cap_list = self.get_cap_list()

        assert len(cap_list) > 0
        cap_list, self.sample_num_frames = self.define_frame_index(cap_list)
        self.lengths = self.sample_num_frames

        n_elements = len(cap_list)
        dataset_prog.set_cap_list(args.dataloader_num_workers, cap_list, n_elements)

        print(f"video length: {len(dataset_prog.cap_list)}", flush=True)

    def set_checkpoint(self, n_used_elements):
        for i in range(len(dataset_prog.n_used_elements)):
            dataset_prog.n_used_elements[i] = n_used_elements

    def __len__(self):
        return dataset_prog.n_elements

    def __getitem__(self, idx):

        data = self.get_data(idx)
        return data

    def get_data(self, idx):
        path = dataset_prog.cap_list[idx]["path"]
        if path.endswith(".mp4"):
            return self.get_video(idx)
        else:
            return self.get_image(idx)

    def get_video(self, idx):
        video_path = dataset_prog.cap_list[idx]["path"]
        assert os.path.exists(video_path), f"file {video_path} do not exist!"
        frame_indices = dataset_prog.cap_list[idx]["sample_frame_index"]
        torchvision_video, _, metadata = torchvision.io.read_video(video_path, output_format="TCHW")
        video = torchvision_video[frame_indices]
        video = self.transform(video)
        video = rearrange(video, "t c h w -> c t h w")
        video = video.to(torch.uint8)
        assert video.dtype == torch.uint8

        h, w = video.shape[-2:]
        assert (
            h / w <= 17 / 16 and h / w >= 8 / 16
        ), f"Only videos with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But video ({video_path}) found ratio is {round(h / w, 2)} with the shape of {video.shape}"

        video = video.float() / 127.5 - 1.0

        text = dataset_prog.cap_list[idx]["cap"]
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)]

        text = text[0] if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_tokens_and_mask["input_ids"]
        cond_mask = text_tokens_and_mask["attention_mask"]
        return dict(
            pixel_values=video,
            text=text,
            input_ids=input_ids,
            cond_mask=cond_mask,
            path=video_path,
        )

    def get_image(self, idx):
        image_data = dataset_prog.cap_list[idx]  # [{'path': path, 'cap': cap}, ...]

        image = Image.open(image_data["path"]).convert("RGB")  # [h, w, c]
        image = torch.from_numpy(np.array(image))  # [h, w, c]
        image = rearrange(image, "h w c -> c h w").unsqueeze(0)  #  [1 c h w]
        # for i in image:
        #     h, w = i.shape[-2:]
        #     assert h / w <= 17 / 16 and h / w >= 8 / 16, f'Only image with a ratio (h/w) less than 17/16 and more than 8/16 are supported. But found ratio is {round(h / w, 2)} with the shape of {i.shape}'

        image = (self.transform_topcrop(image) if "human_images" in image_data["path"] else self.transform(image)
                 )  #  [1 C H W] -> num_img [1 C H W]
        image = image.transpose(0, 1)  # [1 C H W] -> [C 1 H W]

        image = image.float() / 127.5 - 1.0

        caps = (image_data["cap"] if isinstance(image_data["cap"], list) else [image_data["cap"]])
        caps = [random.choice(caps)]
        text = caps
        input_ids, cond_mask = [], []
        text = text[0] if random.random() > self.cfg else ""
        text_tokens_and_mask = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = text_tokens_and_mask["input_ids"]  # 1, l
        cond_mask = text_tokens_and_mask["attention_mask"]  # 1, l
        return dict(
            pixel_values=image,
            text=text,
            input_ids=input_ids,
            cond_mask=cond_mask,
            path=image_data["path"],
        )

    def define_frame_index(self, cap_list):
        new_cap_list = []
        sample_num_frames = []
        cnt_too_long = 0
        cnt_too_short = 0
        cnt_no_cap = 0
        cnt_no_resolution = 0
        cnt_resolution_mismatch = 0
        cnt_movie = 0
        cnt_img = 0
        for i in cap_list:
            path = i["path"]
            cap = i.get("cap", None)
            # ======no caption=====
            if cap is None:
                cnt_no_cap += 1
                continue
            if path.endswith(".mp4"):
                # ======no fps and duration=====
                duration = i.get("duration", None)
                fps = i.get("fps", None)
                if fps is None or duration is None:
                    continue

                # ======resolution mismatch=====
                resolution = i.get("resolution", None)
                if resolution is None:
                    cnt_no_resolution += 1
                    continue
                else:
                    if (resolution.get("height", None) is None or resolution.get("width", None) is None):
                        cnt_no_resolution += 1
                        continue
                    height, width = i["resolution"]["height"], i["resolution"]["width"]
                    aspect = self.max_height / self.max_width
                    hw_aspect_thr = 1.5
                    is_pick = filter_resolution(
                        height,
                        width,
                        max_h_div_w_ratio=hw_aspect_thr * aspect,
                        min_h_div_w_ratio=1 / hw_aspect_thr * aspect,
                    )
                    if not is_pick:
                        print("resolution mismatch")
                        cnt_resolution_mismatch += 1
                        continue

                # import ipdb;ipdb.set_trace()
                i["num_frames"] = math.ceil(fps * duration)
                # max 5.0 and min 1.0 are just thresholds to filter some videos which have suitable duration.
                if i["num_frames"] / fps > self.video_length_tolerance_range * (
                        self.num_frames / self.train_fps *
                        self.speed_factor):  # too long video is not suitable for this training stage (self.num_frames)
                    cnt_too_long += 1
                    continue

                # resample in case high fps, such as 50/60/90/144 -> train_fps(e.g, 24)
                frame_interval = fps / self.train_fps
                start_frame_idx = 0
                frame_indices = np.arange(start_frame_idx, i["num_frames"], frame_interval).astype(int)

                # comment out it to enable dynamic frames training
                if (len(frame_indices) < self.num_frames and random.random() < self.drop_short_ratio):
                    cnt_too_short += 1
                    continue

                #  too long video will be temporal-crop randomly
                if len(frame_indices) > self.num_frames:
                    begin_index, end_index = self.temporal_sample(len(frame_indices))
                    frame_indices = frame_indices[begin_index:end_index]
                    # frame_indices = frame_indices[:self.num_frames]  # head crop
                i["sample_frame_index"] = frame_indices.tolist()
                new_cap_list.append(i)
                i["sample_num_frames"] = len(i["sample_frame_index"])  # will use in dataloader(group sampler)
                sample_num_frames.append(i["sample_num_frames"])
            elif path.endswith(".jpg"):  # image
                cnt_img += 1
                new_cap_list.append(i)
                i["sample_num_frames"] = 1
                sample_num_frames.append(i["sample_num_frames"])
            else:
                raise NameError(
                    f"Unknown file extension {path.split('.')[-1]}, only support .mp4 for video and .jpg for image")
        # import ipdb;ipdb.set_trace()
        main_print(
            f"no_cap: {cnt_no_cap}, too_long: {cnt_too_long}, too_short: {cnt_too_short}, "
            f"no_resolution: {cnt_no_resolution}, resolution_mismatch: {cnt_resolution_mismatch}, "
            f"Counter(sample_num_frames): {Counter(sample_num_frames)}, cnt_movie: {cnt_movie}, cnt_img: {cnt_img}, "
            f"before filter: {len(cap_list)}, after filter: {len(new_cap_list)}")
        return new_cap_list, sample_num_frames

    def decord_read(self, path, frame_indices):
        decord_vr = self.v_decoder(path)
        video_data = decord_vr.get_batch(frame_indices).asnumpy()
        video_data = torch.from_numpy(video_data)
        video_data = video_data.permute(0, 3, 1, 2)  # (T, H, W, C) -> (T C H W)
        return video_data

    def read_jsons(self, data):
        cap_lists = []
        with open(data, "r") as f:
            folder_anno = [i.strip().split(",") for i in f.readlines() if len(i.strip()) > 0]
        print(folder_anno)
        for folder, anno in folder_anno:
            with open(anno, "r") as f:
                sub_list = json.load(f)
            for i in range(len(sub_list)):
                sub_list[i]["path"] = opj(folder, sub_list[i]["path"])
            cap_lists += sub_list
        return cap_lists

    def get_cap_list(self):
        cap_lists = self.read_jsons(self.data)
        return cap_lists
