import json

# [1, 24, 46336, 128]

# q =  torch.randn(1, 24, 46336, 128)

# for startegy (2, 6, 1)
# theortical speed up is 10.0
# actual speed up is 7.6770869514877855
# for startegy (1, 6, 10)
# theortical speed up is 2.0
# actual speed up is 1.81064261469606
# for startegy (2, 3, 3)
# theortical speed up is 6.666666666666667
# actual speed up is 5.410431320071153
# for startegy (2, 6, 10)
# theortical speed up is 1.0
# actual speed up is 0.9610114048271594
# for startegy (2, 1, 10)
# theortical speed up is 6.0
# actual speed up is 5.1464607264824584
# for startegy (2, 3, 5)
# theortical speed up is 4.0
# actual speed up is 3.5396014498556014


initial_path = "/workspace/codefolder/FastVideo/assets/mask_strategy_hunyuan.json"

actuall_speed_up = 0

with open(initial_path, "r") as f:
    data = json.load(f)
    for key in data.keys():
        t, h, w = data[key]
        data[key] = [min(2,t), h, w]
        
        t, h, w = data[key]
        
        if t == 2 and h == 6 and w == 1:
            actuall_speed_up += 7.5
        elif t == 1 and h == 6 and w == 10:
            actuall_speed_up += 1.77
        elif t == 2 and h == 3 and w == 3:
            actuall_speed_up += 5.32
        elif t == 2 and h == 6 and w == 10:
            actuall_speed_up += 0.94
        elif t == 2 and h == 1 and w == 10:
            actuall_speed_up += 5.07
        elif t == 2 and h == 3 and w == 5:
            actuall_speed_up += 3.47

print(actuall_speed_up/len(data.keys()))

save_path = "/workspace/codefolder/FastVideo/assets/test_mask_strategy_hunyuan.json"


# print unique values for new data
unique_values = set()
for key in data.keys():
    unique_values.add(tuple(data[key]))
print(unique_values)

with open(save_path, "w") as f:
    json.dump(data, f, indent=4)