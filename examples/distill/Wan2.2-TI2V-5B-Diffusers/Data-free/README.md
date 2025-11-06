# Wan2.2-5B Distill Example
These are end-to-end example scripts for distilling Wan2.2 TI2V 5B model DMD+VSA methods.

### 0. Make sure you have installed VSA

```bash
pip install vsa
```

### Data-free Distillation
When `--simulate_generator_forward` is enabled, distillation becomes data-free by simulating intermediate steps through forward inference of the generator. This helps avoid training–inference mismatch. See Section 4.5 of [DMD2](https://arxiv.org/pdf/2405.14867) for details.