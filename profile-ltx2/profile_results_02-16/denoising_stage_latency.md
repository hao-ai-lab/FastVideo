# Setting: 
- 4 GPUS
- VSA Sparsity: 0.7
- Theoretical Speedup: 3.3x
- Dimensions: 121 * 1920 * 1088 (32640 tokens on FA, 34560 tokens on VSA)

# Denoising Stage Latency (Using stage logging)

Denoising stage: `LTX2DenoisingStage` (ms)

| Metric | `profile.4.compile.log` (ms) | `profile.4.vsa.compile.log` (ms) |
|---|---:|---:|
| run_01 | 31198.481446132064 | 16856.768134981394 |
| run_02 | 7174.036046490073 | 5731.170237064362 |
| run_03 | 7180.428974330425 | 5779.2842369526625 |
| run_04 | 7331.019859761 | 5753.65149974823 |
| run_05 | 7166.119007393718 | 5824.750360101461 |
| run_06 | 7196.353662759066 | 5737.696401774883 |
| run_07 | 7207.990800961852 | 5748.391387984157 |
| run_08 | 7205.886080861092 | 5744.493938982487 |
| run_09 | 7176.073379814625 | 5711.566410958767 |
| run_10 | 7169.360462576151 | 5760.473757982254 |
| avg_all_rounds | 9600.57497 | 6864.82464 |
| avg_excl_warmup | 7200.80759 | 5754.60869 |


# Attention E2E time breakdown: (Using perf counter)
  - VSA avg: 9.534198 ms
    - shape: 34560 * 8 * 128
    - Total time: 9.534198 * 48 * 8 = 3,661.132032 ms
    - percentage: 3661.132032 / 5754.60869 * 100 = 63.62%
  - FA avg: 12.999318 ms
    - shape: 32640 * 8 * 128
    - Total time: 12.999318 * 48 * 8 = 4,991.738112 ms
    - percentage: 4991.738112 / 7200.80759 * 100 = 69.32%




# Attention kernel time (From trace files):
- VSA: 
    - Single kernel: 2.762085 ms 
    - .forward(): 4.002 ms
    - total (including preprocess qkv): 5.513 ms
- FA: 9.534580 ms
    - .forward() (single kernel):9.534580 ms 



