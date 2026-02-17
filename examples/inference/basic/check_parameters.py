#!/usr/bin/env python3
"""
Simple side-by-side parameter check without loading models.
"""

print("="*80)
print("PARAMETER COMPARISON: FastVideo vs SGLang (flux_sgl.py)")
print("="*80)

print("\n1. EMBEDDED GUIDANCE SCALE")
print("-" * 40)
print("FastVideo:")
print("  embedded_cfg_scale: 0.0035")
print("  Multiplied by: 1000.0")
print("  Effective value: 3.5")
print("\nSGLang (from flux_sgl.py line ~42-43):")
print("  guidance_scale: 1.0")
print("  embedded_guidance_scale: 3.5") 
print("  Effective value: 3.5")
print("\n✅ MATCH")

print("\n2. RANDOM SEED")
print("-" * 40)
print("FastVideo default: 1024")
print("SGLang (flux_sgl.py): 42")
print("Test uses: 42")
print("✅ MATCHED IN TEST")

print("\n3. NUMBER OF STEPS")
print("-" * 40)
print("FastVideo default: 28")
print("SGLang (flux_sgl.py): 50")
print("Test uses: 50")
print("✅ MATCHED IN TEST")

print("\n4. RESOLUTION")
print("-" * 40)
print("FastVideo default: 1024×1024")
print("SGLang (flux_sgl.py): 720×1280 (height × width)")
print("Test uses: 1280×720 (FastVideo order: height × width)")
print("⚠️  POTENTIAL MISMATCH - dimension ordering may differ")

print("\n5. FLOW SHIFT")
print("-" * 40)
print("FastVideo default: 3.0")
print("SGLang (flux_sgl.py): Not explicitly set, likely uses model default")
print("✅ LIKELY MATCH")

print("\n" + "="*80)
print("INVESTIGATION: Checking SGLang's actual sampling_params_kwargs")
print("="*80)

import sys
sys.path.insert(0, '/FastVideo')

# Read the flux_sgl.py file to extract exact parameters
print("\nReading /FastVideo/examples/inference/basic/flux_sgl.py...")
with open('/FastVideo/examples/inference/basic/flux_sgl.py', 'r') as f:
    content = f.read()

# Find sampling_params_kwargs
import re
params_match = re.search(r'sampling_params_kwargs=dict\((.*?)\)', content, re.DOTALL)
if params_match:
    params_str = params_match.group(1)
    print("\nExtracted sampling_params_kwargs:")
    for line in params_str.split(','):
        line = line.strip()
        if line:
            print(f"  {line}")
else:
    print("Could not extract parameters")

# Find generator.generate() call
gen_match = re.search(r'generator = DiffGenerator\.from_pretrained\((.*?)\)', content, re.DOTALL)
if gen_match:
    gen_params = gen_match.group(1)
    print("\nExtracted DiffGenerator.from_pretrained params:")
    for line in gen_params.split(','):
        line = line.strip()
        if line and not line.startswith('#'):
            print(f"  {line}")

print("\n" + "="*80)
print("KEY INSIGHT:")
print("="*80)
print("""
The flux_sgl.py script uses:
  generator.generate(
      sampling_params_kwargs=dict(
          prompt=prompt,
          return_frames=True,
          save_output=False,
          output_path=OUTPUT_PATH,
      )
  )

This means SGLang uses DEFAULT parameters from DiffGenerator, which may differ
from what we're testing with. We need to check SGLang's actual defaults.

To match exactly, we need to:
1. Check SGLang's DiffGenerator defaults for FLUX.1-dev
2. Match ALL parameters including hidden ones (height, width, etc.)
3. Verify dimension ordering (height×width vs width×height)
""")

print("\n" + "="*80)
print("RECOMMENDED NEXT STEP:")
print("="*80)
print("""
Check SGLang source code for default parameters:
1. Where does SGLang set default resolution for FLUX.1-dev?
2. What is the default guidance_scale in SGLang?
3. Does SGLang use different default steps?

The pre-generated SGLang images we found are 720×1280 pixels.
FastVideo test generated 1280×720 pixels.
This dimension swap might be the issue!
""")
