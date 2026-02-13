import json
import os
import shutil

import cv2

train = "zelda"

if train == "zelda":
    height = 480
    width = 832
    num_frames = 81
    action_dir = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/actions_81"
elif train == "mc":
    height = 352
    width = 640
    num_frames = 77
    action_dir = "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/actions"
else:
    raise ValueError(f"Invalid train type: {train}")

# Output path
output_path = (
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/"
    f"WanGame2.1_1.3b_i2v/validation_{train}.json"
)

# Fixed fields
fixed_fields = {
    "video_path": None,
    "num_inference_steps": 40,
    "height": height,
    "width": width,
    "num_frames": num_frames,
}

# WASDudlr: single key W.npy, single camera u.npy, key+camera w_u.npy
still = os.path.join(action_dir, "still.npy")
key_W = os.path.join(action_dir, "W.npy")
key_S = os.path.join(action_dir, "S.npy")
key_A = os.path.join(action_dir, "A.npy")
key_D = os.path.join(action_dir, "D.npy")
key_wa = os.path.join(action_dir, "WA.npy")
key_s_u = os.path.join(action_dir, "S_u.npy")
camera_u = os.path.join(action_dir, "u.npy")
camera_d = os.path.join(action_dir, "d.npy")
camera_l = os.path.join(action_dir, "l.npy")
camera_r = os.path.join(action_dir, "r.npy")
# key_1_action_rand, camera_1_action_rand (full set); _f4 suffix for granularity 4
key_1_action_rand_1 = os.path.join(action_dir, "key_1_action_rand_1.npy")
key_1_action_rand_2 = os.path.join(action_dir, "key_1_action_rand_2.npy")
key_1_action_rand_1_f4 = os.path.join(action_dir, "key_1_action_rand_1_f4.npy")
key_1_action_rand_2_f4 = os.path.join(action_dir, "key_1_action_rand_2_f4.npy")
camera_1_action_rand_1 = os.path.join(action_dir, "camera_1_action_rand_1.npy")
camera_1_action_rand_2 = os.path.join(action_dir, "camera_1_action_rand_2.npy")
camera_1_action_rand_1_f4 = os.path.join(action_dir, "camera_1_action_rand_1_f4.npy")
camera_1_action_rand_2_f4 = os.path.join(action_dir, "camera_1_action_rand_2_f4.npy")
key_camera_1_action_rand_1 = os.path.join(action_dir, "key_camera_1_action_rand_1.npy")
key_camera_1_action_rand_2 = os.path.join(action_dir, "key_camera_1_action_rand_2.npy")
key_camera_1_action_rand_1_f4 = os.path.join(action_dir, "key_camera_1_action_rand_1_f4.npy")
key_camera_1_action_rand_2_f4 = os.path.join(action_dir, "key_camera_1_action_rand_2_f4.npy")
key_camera_excl_1_action_rand_1 = os.path.join(action_dir, "key_camera_excl_1_action_rand_1.npy")
key_camera_excl_1_action_rand_2 = os.path.join(action_dir, "key_camera_excl_1_action_rand_2.npy")
key_camera_excl_1_action_rand_1_f4 = os.path.join(action_dir, "key_camera_excl_1_action_rand_1_f4.npy")
key_camera_excl_1_action_rand_2_f4 = os.path.join(action_dir, "key_camera_excl_1_action_rand_2_f4.npy")
# key_2_action_rand, camera_2_action_rand (WASD/UDLR+still)
key_2_action_rand_1 = os.path.join(action_dir, "key_2_action_rand_1.npy")
key_2_action_rand_1_f4 = os.path.join(action_dir, "key_2_action_rand_1_f4.npy")
camera_2_action_rand_1 = os.path.join(action_dir, "camera_2_action_rand_1.npy")
camera_2_action_rand_1_f4 = os.path.join(action_dir, "camera_2_action_rand_1_f4.npy")
key_camera_2_action_rand_1 = os.path.join(action_dir, "key_camera_2_action_rand_1.npy")
key_camera_2_action_rand_1_f4 = os.path.join(action_dir, "key_camera_2_action_rand_1_f4.npy")
key_camera_excl_2_action_rand_1 = os.path.join(action_dir, "key_camera_excl_2_action_rand_1.npy")
key_camera_excl_2_action_rand_1_f4 = os.path.join(action_dir, "key_camera_excl_2_action_rand_1_f4.npy")


train_img_zelda_list = [
    # "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/-BxyBxfDKA0_chunk_0292/segment0001.jpg",
    # "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/-BxyBxfDKA0_chunk_0292/segment0003.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/5TTrlqAguhQ_chunk_0006/segment0002.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/5TTrlqAguhQ_chunk_0067/segment0002.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/5TTrlqAguhQ_chunk_0484/segment0002.jpg",  
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/N6ObBAt41bg_chunk_0019/segment0004.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/N6ObBAt41bg_chunk_0140/segment0003.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/zelda/N6ObBAt41bg_chunk_0300/segment0003.jpg",
]

val_img_zelda_list = train_img_zelda_list
train_action_zelda_list = []
for img in train_img_zelda_list:
    img_dir = os.path.dirname(img)
    basename = os.path.splitext(os.path.basename(img))[0]
    action_path = os.path.join(
        img_dir,
        "postprocess/action/majority_voting/"
        "81_frame_no_button",
        f"{basename}.npy",
    )
    train_action_zelda_list.append(action_path)


val_img_mc_list = [
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000002.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000003.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000004.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000005.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000000.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000001.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000006.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/mc_wasd_10/validate/000007.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/humanplay/000005.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/humanplay/000013.jpg",
]

# Get train data list
train_mc_data_dir = "/mnt/weka/home/hao.zhang/mhuo/traindata_0208_2000/data/wasd4holdrandview_simple_1key1mouse1"
train_mc_idx_list = ["000000", "000500", "001000", "001500", "002000", "002500", "003000", "003500"]
train_mc_img_list = []
train_mc_action_list = []

for idx in train_mc_idx_list:
    video_path = os.path.join(train_mc_data_dir, f"videos/{idx}.mp4")
    # extract the first frame as image
    image_path = os.path.join(train_mc_data_dir, f"first_frame/{idx}.jpg")
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        cv2.imwrite(image_path, frame)
        train_mc_img_list.append(image_path)
        train_mc_action_list.append(os.path.join(train_mc_data_dir, f"videos/{idx}_action.npy"))


# Get doom Val data list
val_img_doom_list = [
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/doom/000000.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/doom/000001.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/doom/000002.jpg",
    "/mnt/weka/home/hao.zhang/mhuo/FastVideo/examples/training/finetune/WanGame2.1_1.3b_i2v/doom/000003.jpg",
]

if train == "mc":
    val_img_list = val_img_mc_list
    train_img_list = train_mc_img_list
    train_action_list = train_mc_action_list
elif train == "zelda":
    val_img_list = val_img_zelda_list
    train_img_list = train_img_zelda_list
    train_action_list = train_action_zelda_list
elif train == "doom":
    val_img_list = val_img_doom_list
else:
    raise ValueError(f"Invalid train type: {train}")


holder = 0 # placeholder
# 32 placeholders (idx 0-31). Fill in manually.
a0 = ["00 Val-00: W", val_img_list[0], key_W]
a1 = ["01 Val-01: S", val_img_list[1], key_S]
a2 = ["02 Val-02: A", val_img_list[2], key_A]
a3 = ["03 Val-03: D", val_img_list[3], key_D]
a4 = ["04 Val-04: u", val_img_list[4], camera_u]
a5 = ["05 Val-05: d", val_img_list[5], camera_d]
a6 = ["06 Val-06: l", val_img_list[4], camera_l]
a7 = ["07 Val-07: r", val_img_list[5], camera_r]
a8 = ["08 Val-00: key rand", val_img_list[0], key_1_action_rand_1]
a9 = ["09 Val-01: key rand", val_img_list[1], key_1_action_rand_2]
a10 = ["10 Val-02: camera rand", val_img_list[2], camera_1_action_rand_1]
a11 = ["11 Val-03: camera rand", val_img_list[3], camera_1_action_rand_2]
a12 = ["12 Val-00: key+camera excl rand", val_img_list[0], key_camera_excl_1_action_rand_1]
a13 = ["13 Val-01: key+camera excl rand", val_img_list[1], key_camera_excl_1_action_rand_2]
a14 = ["14 Val-02: key+camera rand", val_img_list[2], key_camera_1_action_rand_1]
a15 = ["15 Val-03: key+camera rand", val_img_list[3], key_camera_1_action_rand_2]
a16 = ["16 Val-04: (simultaneous) key rand", val_img_list[4], key_2_action_rand_1]
a17 = ["17 Val-05: (simultaneous) camera rand", val_img_list[5], camera_2_action_rand_1]
a18 = ["18 Val-06: (simultaneous) key+camera excl rand", val_img_list[5], key_camera_excl_2_action_rand_1]
a19 = ["19 Val-07: (simultaneous) key+camera rand", val_img_list[5], key_camera_2_action_rand_1]
a20 = ["20 Val-08: W+A", val_img_list[0], key_wa]
a21 = ["21 Val-09: S+u", val_img_list[1], key_s_u]
a22 = ["22 Val-08: Still", val_img_list[2], still]
a23 = ["23 Val-09: Still", val_img_list[3], still]
a24 = ["24 Val-06: key+camera excl rand Frame 4", val_img_list[4], key_camera_excl_1_action_rand_1_f4]
a25 = ["25 Val-07: key+camera excl rand Frame 4", val_img_list[5], key_camera_excl_1_action_rand_2_f4]
a26 = ["26 Train-00", train_img_list[0], train_action_list[0]]
a27 = ["27 Train-01", train_img_list[1], train_action_list[1]]
# a28 = ["28 Train-02", train_img_list[2], train_action_list[2]]
# a29 = ["29 Train-03", train_img_list[3], train_action_list[3]]
# a30 = ["30 Train-04", train_img_list[4], train_action_list[4]]
# a31 = ["31 Train-05", train_img_list[5], train_action_list[5]]
a28 = ["28 Doom-00: W", val_img_doom_list[0], key_W]
a29 = ["29 Doom-01: key rand", val_img_doom_list[1], key_1_action_rand_1]
a30 = ["30 Doom-02: camera rand", val_img_doom_list[2], camera_1_action_rand_1]
a31 = ["31 Doom-03: key+camera excl rand", val_img_doom_list[3], key_camera_excl_1_action_rand_1]

Val_entries = {
    0: a0,
    1: a1,
    2: a2,
    3: a3,
    4: a4,
    5: a5,
    6: a6,
    7: a7,
    8: a8,
    9: a9,
    10: a10,
    11: a11,
    12: a12,
    13: a13,
    14: a14,
    15: a15,
    16: a16,
    17: a17,
    18: a18,
    19: a19,
    20: a20,
    21: a21,
    22: a22,
    23: a23,
    24: a24,
    25: a25,
    26: a26,
    27: a27,
    28: a28,
    29: a29,
    30: a30,
    31: a31,
}

data = []
for idx in range(32):
    if idx not in Val_entries:
        raise ValueError(f"Missing entry for idx {idx}")
    caption, image_path, action_path = Val_entries[idx]
    data.append(
        {
            "caption": caption,
            "image_path": image_path,
            "action_path": action_path,
            **fixed_fields,
        }
    )

output = {"data": data}
with open(output_path, "w") as f:
    json.dump(output, f, indent=4)

print(f"Generated {len(data)} entries to {output_path}")

# Check file all exists

with open(output_path) as f:
    data = json.load(f)

missing = []
for i, item in enumerate(data['data']):
    for key in ('image_path', 'action_path'):
        path = item.get(key)
        if path:
            import os
            if not os.path.isfile(path):
                missing.append((i, key, path))
if missing:
    print('Missing paths:')
    for idx, key, path in missing:
        print(f'  [{idx}] {key}: {path}')
else:
    print('All paths exist.')


