#!/usr/bin/env python3
import os
import runpy
import sys

base = os.environ["TORCHINDUCTOR_CACHE_DIR_BASE"]
local_rank = os.environ["LOCAL_RANK"]
os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(base, f"rank{local_rank}")

script = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(script, run_name="__main__")
