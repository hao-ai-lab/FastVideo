# Step 1 — KD training                                                                                      
  bash examples/train/run.sh examples/train/configs/kdtest/kd_causal.yaml                                     
                                                                                                              
  # Step 2 — Export ode_init                                                                                  
python -m fastvideo.train.entrypoint.dcp_to_diffusers --role student --checkpoint outputs/kdtest/kd_causal/checkpoint-300 --output-dir outputs/kdtest/kd_ode_init
                                                                                                              
  # Step 3 — Self-Forcing                                                                                     
  bash examples/train/run.sh examples/train/configs/kdtest/self_forcing_causal.yaml