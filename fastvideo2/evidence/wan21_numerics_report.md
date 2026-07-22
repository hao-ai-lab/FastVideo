# Wan2.1 numerics vs official implementation

Goldens: `https://github.com/Wan-Video/Wan2.1` @ `9737cba9c1c3c4d04b33fcad41c111989865d315` — see `manifest.json`.
Cells are relative L2 vs the official output (lower is closer; `FAIL` = over tolerance).

| probe | fastvideo2 | fastvideo_main |
|---|---|---|
| text_encoder.prompt | 0.00e+00 | 0.00e+00 |
| text_encoder.negative | 0.00e+00 | **FAIL** 4.19e-01 |
| dit.t750 | 1.59e-02 | 1.68e-02 |
| dit.t250 | 1.99e-02 | 2.12e-02 |
| vae.decode | 2.99e-04 | 2.99e-04 |
