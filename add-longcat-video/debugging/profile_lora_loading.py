#!/usr/bin/env python3
"""
Profile where time is spent during LoRA loading.
"""

import time
import torch
from fastvideo import VideoGenerator

def timed_section(name):
    """Context manager to time sections."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"\n[START] {name}...")
            return self
        
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"[DONE] {name} took {elapsed:.2f}s")
    
    return Timer()

print("=" * 80)
print("Profiling LoRA Loading Time")
print("=" * 80)

with timed_section("Loading base model WITHOUT LoRA"):
    generator = VideoGenerator.from_pretrained(
        "weights/longcat-native",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
    )

print("\n" + "=" * 80)
print("Now loading LoRA adapter (this is the slow part)")
print("=" * 80)

with timed_section("set_lora_adapter() call"):
    generator.set_lora_adapter(
        lora_nickname="distilled",
        lora_path="weights/longcat-native/lora/cfg_step_lora.safetensors"
    )

print("\n" + "=" * 80)
print("SUCCESS! LoRA loaded.")
print("=" * 80)

print("\nNow let's test loading WITH lora_path from the start...")
generator.shutdown()
time.sleep(2)

print("\n" + "=" * 80)
print("Loading model WITH lora_path (should be same speed as without)")
print("=" * 80)

with timed_section("from_pretrained WITH lora_path"):
    generator2 = VideoGenerator.from_pretrained(
        "weights/longcat-native",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        lora_path="weights/longcat-native/lora/cfg_step_lora.safetensors",
        lora_nickname="distilled"
    )

print("\n" + "=" * 80)
print("Profiling Complete")
print("=" * 80)


"""
Profile where time is spent during LoRA loading.
"""

import time
import torch
from fastvideo import VideoGenerator

def timed_section(name):
    """Context manager to time sections."""
    class Timer:
        def __enter__(self):
            self.start = time.time()
            print(f"\n[START] {name}...")
            return self
        
        def __exit__(self, *args):
            elapsed = time.time() - self.start
            print(f"[DONE] {name} took {elapsed:.2f}s")
    
    return Timer()

print("=" * 80)
print("Profiling LoRA Loading Time")
print("=" * 80)

with timed_section("Loading base model WITHOUT LoRA"):
    generator = VideoGenerator.from_pretrained(
        "weights/longcat-native",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
    )

print("\n" + "=" * 80)
print("Now loading LoRA adapter (this is the slow part)")
print("=" * 80)

with timed_section("set_lora_adapter() call"):
    generator.set_lora_adapter(
        lora_nickname="distilled",
        lora_path="weights/longcat-native/lora/cfg_step_lora.safetensors"
    )

print("\n" + "=" * 80)
print("SUCCESS! LoRA loaded.")
print("=" * 80)

print("\nNow let's test loading WITH lora_path from the start...")
generator.shutdown()
time.sleep(2)

print("\n" + "=" * 80)
print("Loading model WITH lora_path (should be same speed as without)")
print("=" * 80)

with timed_section("from_pretrained WITH lora_path"):
    generator2 = VideoGenerator.from_pretrained(
        "weights/longcat-native",
        num_gpus=1,
        dit_cpu_offload=False,
        vae_cpu_offload=True,
        text_encoder_cpu_offload=True,
        lora_path="weights/longcat-native/lora/cfg_step_lora.safetensors",
        lora_nickname="distilled"
    )

print("\n" + "=" * 80)
print("Profiling Complete")
print("=" * 80)





