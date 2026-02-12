# Waypoint-1-Small in FastVideo — Quick start

Three ways to run the model, from “I just want a clip” to “I want a live window”.

---

## 1. RunPod / cloud (no local GPU) — **what you already did**

Use this when you only have SSH (e.g. RunPod) and want a short video file.

1. SSH into your pod and go to the repo:
   ```bash
   cd ~/FastVideo
   ```
2. Run the batch script (generates 8 frames with “hold W”, saves MP4):
   ```bash
   python examples/inference/basic/basic_waypoint.py
   ```
3. Download the video (on your PC):
   ```powershell
   scp -P 22010 -i $env:USERPROFILE\.ssh\id_ed25519 root@YOUR_POD_IP:~/FastVideo/video_samples_waypoint_runpod/waypoint_runpod.mp4 .
   ```

Options: `--num_steps 12`, `--height 368`, `--width 640`, `--seed 42`, `--prompt "Your prompt"`.

---

## 2. Local GPU — terminal interactive (type keys, then get video)

Use this when you have an **NVIDIA GPU on your PC** and want to type W/A/S/D in the terminal and get a saved video at the end.

```bash
cd FastVideo
python examples/inference/basic/basic_waypoint_streaming.py
```

- You’ll be prompted: type `w`, `a`, `s`, `d`, `wa`, etc., then Enter. Each line runs one step.
- Type `q` and Enter to quit; the script saves the clip to `video_samples_waypoint/`.

---

## 3. Local GPU — live window (press keys, see frame update)

Use this when you have an **NVIDIA GPU on your PC** and want a window where you press W/A/S/D and see the next frame appear.

```bash
cd FastVideo
python examples/inference/basic/basic_waypoint_local_live.py
```

- A window opens. Press **W / A / S / D / Space** to move; each keypress runs one generation step and updates the window.
- Press **Q** or **Esc** to quit.
- To save the clip when you quit:
  ```bash
  python examples/inference/basic/basic_waypoint_local_live.py --save_on_quit --output_path my_clip.mp4
  ```

Note: each step takes 1–3 seconds on a typical GPU, so it won’t feel like 30 FPS unless you have a very fast card.

---

## Summary

| Goal                         | Where to run | Script                          |
|-----------------------------|--------------|----------------------------------|
| Get a short clip (batch)   | RunPod / SSH | `basic_waypoint.py`             |
| Type keys, save video      | Local GPU    | `basic_waypoint_streaming.py`    |
| Window + keys, see frames  | Local GPU    | `basic_waypoint_local_live.py`   |

If you don’t have a local GPU, use **1** (RunPod). If you do have a local GPU and want the “game-like” feel, use **3** (`basic_waypoint_local_live.py`).
