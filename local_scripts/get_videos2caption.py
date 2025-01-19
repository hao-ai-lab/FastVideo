import json
src_json = "../HD-MixKit-Hunyuan-Distill-125x768x1280/videos2caption.json"
des_json = "./videos2caption.json"
syn_num = 8
with open(src_json) as f:
    data = json.load(f)
data = data[:syn_num]
# from IPython import embed
# embed()

with open(des_json, "w") as f2:
    json.dump(data, f2, indent=4, ensure_ascii=False)