# Fast Video
This is currently based on Open-Sora-1.2.0: https://github.com/PKU-YuanGroup/Open-Sora-Plan/tree/294993ca78bf65dec1c3b6fb25541432c545eda9

## Envrironment
Change the index-url cuda version according to your system.
```
conda create -n fastvideo python=3.10.12
conda activate fastvideo
pip3 install torch==2.5.0 torchvision  --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/huggingface/diffusers.git@76b7d86a9a5c0c2186efa09c4a67b5f5666ac9e3
pip install packaging ninja && pip install flash-attn==2.7.0.post2 --no-build-isolation 
```

```
pip install -e . && pip install -e ".[train]"
sudo apt-get update && apt install screen && pip install watch gpustat
```

## Prepare Data & Models
We've prepared some debug data to facilitate development. To make sure the training pipeline is correct, train on the debug data and make sure the model overfit on it (feed it the same text prompt and see if the output video is the same as the training data)

```
mkdir data && mkdir data/outputs/
python scripts/download_hf.py --repo_id=Stealths-Video/mochi_diffuser --local_dir=data/mochi --repo_type=model
python scripts/download_hf.py --repo_id=Stealths-Video/Merge-30k-Data --local_dir=data/Merge-30k-Data --repo_type=dataset
python scripts/download_hf.py --repo_id=Stealths-Video/validation_embeddings --local_dir=data/validation_embeddings --repo_type=dataset
cd data/Merge-30k-Data
cat Merged30K.tar.gz.part.* > Merged30K.tar.gz
rm Merged30K.tar.gz.part.*
tar --use-compress-program="pigz --processes 64" -xvf Merged30K.tar.gz
mv ephemeral/hao.zhang/codefolder/FastVideo-OSP/data/Merged-30K-Data/* .
rm -r ephemeral
rm Merged30K.tar.gz
cd ../..
```

## Things Learned 
1. shift8 clear but got structural artifacts
2. lq, 0.025 vague
3. adv not really helpful
4. shift8 euler steps 50 v.s. 100 very similar 
5.  为啥image不会越distill越炸
6. EMA, 大batchsize, 1.5,2.5,3.5,4.5
7. Must have schedule
8. phase 1, 2 learning rate 5e-6不行

## Experiments
Scripts are located at scripts/experiment_N.sh

1. pcm_linear_quadratic， euler_steps 50, 0.025
2. pcm_linear_quadratic， euler_steps 50, 0.05
3. shift 8, euler_steps 100
4. shift 8, euler_steps 50
5. shift 8, euler_steps 100, adv
6. pcm_linear_quadratic， euler_steps 50, 0.025, adv
7. pcm_linear_quadratic， euler_steps 50, 0.05, multiphase 125
8. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75
9. pcm_linear_quadratic， euler_steps 50, 0.05, range 0.75
10. pcm_linear_quadratic， euler_steps 50, 0.05, batchsize 32
11. pcm_linear_quadratic， euler_steps 50, learning rate,1e-7
12. shift1, euler_steps 50

13. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 1
14. 4.5 cfg, validation no cfg, pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75
15. pcm_linear_quadratic， euler_steps 50, 0.15, linear_range 0.75
16. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75  ema 0.95, decay 0.0 


17. no cfg, validation no cfg, pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75
18. shift16, euler_steps 50

19. 4step_infer_shift16_euler_50
20. 4step_infer_shift12_euler_50
21. 4step_infer_lq_euler_50_thresh0.1_lrg_0.75
22. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 1, lr 1e-7
23. lq_euler_50_thres0.1_lrg_0.75_bs_64
24. lq_euler_50_thres0.1_lrg_0.75_lr5e-7



25. shift1_euler_50_0.75_phase1
26. kill
27. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 1, ema 0.95, cfg 4.5

28. lq_euler_50_thresh0.1_lrg_0.75_phase1_ema0.95
29. lq_euler_50_thres0.1_lrg_0.75_phase_ema0.95_cfg7
30. lq_euler_50_thresh0.1_lrg_0.75_phase1_ema0.98_cfg4.5
31. lq_euler_50_thresh0.1_lrg_0.75_phase1_lr_3e-7
32. lq_euler_50_thresh0.15_lrg_0.75_phase1_ema0.95_cfg4.5
33. lq_euler_50_thres0.1_linear_range_0.75_repro
34. lq_euler_50_thres0.1_lrg_0.75_reproduc

35. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 1, learning rate 5e-6
36. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 2, learning rate 1e-6
37. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 2, learning rate 5e-6
38. lq_euler_50_thres0.1_linear_range_0.75, learning rate 5e-6
39. lq_euler_50_thres0.1_linear_range_0.75, learning rate 1e-5
40. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 1, learning rate 1e-6


41. lq_euler_50_thres0.1_lrg_0.75_reproduce
42. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 4, learning rate 1e-6
43. pcm_linear_quadratic， euler_steps 50, 0.1, linear_range 0.75, phase 1, learning rate 1e-6, cfg 6.0
44. lq_euler_50_thres0.1_lrg_0.75_phase1_lr_5e-6_test_norm
45. lq_euler_50_thres0.1_lrg_0.75_phase1_lr_5e-6_pred_decay_0.1_latent14
46. lq_euler_50_thres0.1_lrg_0.75_phase1_lr1e-6_pred_decay0.1
47. lq_euler_50_thres0.1_lrg_0.75_phase1_lr1e-6_pred_decay0.05
48. lq_euler_50_thres0.1_lrg_0.75_phase1_lr1e-6_pred_decay0.01