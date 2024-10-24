# Fast Video

## Envrironment
We currrently we CUDA 11.8 + torch 2.1.0.

(Upgrade this when we need flexatteniton)

```
pip install -e .
pip install -e ".[train]"
```

## Training
```
bash scripts/text_condition/gpu/train_t2v.sh
```


## TODO

- [ ] Delete all npu related stuff.
- [ ] Create dummy debug data.
- [ ] Add Mochi
- [ ] Add Mochi VAE
