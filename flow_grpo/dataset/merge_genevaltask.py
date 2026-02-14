import json
import random
from collections import defaultdict

def distribute_samples(tasks, weights, total_samples):
    # 计算每个任务的浮点样本数
    float_samples = {task: weights[i] * total_samples for i, task in enumerate(tasks)}
    # 取整数部分
    int_samples = {task: int(fs) for task, fs in float_samples.items()}
    remainder = total_samples - sum(int_samples.values())
    
    # 计算小数部分并按大小排序
    fractional_parts = {task: float_samples[task] - int_samples[task] for task in tasks}
    sorted_tasks = sorted(tasks, key=lambda x: fractional_parts[x], reverse=True)
    
    # 分配余数到前remainder个任务
    for task in sorted_tasks[:remainder]:
        int_samples[task] += 1
    
    return int_samples

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def sample_data(data, samples_needed):
    if len(data) >= samples_needed:
        return random.sample(data, samples_needed)
    else:
        return random.choices(data, k=samples_needed)

def merge_datasets_with_weights(tasks, weights, output_path, total_samples=50000):
    # 确保权重和任务数量一致且总和为1
    assert len(tasks) == len(weights), "Tasks and weights must have the same length"
    
    # 将权重转换为字典格式
    samples_per_task = distribute_samples(tasks, weights, total_samples)
    print(samples_per_task)
    all_sampled_data = []
    
    for task in tasks:
        file_path = f'dataset/geneval_ood_60_20/{task}/evaluation_metadata.jsonl'
        data = read_jsonl(file_path)
        samples_needed = samples_per_task[task]
        
        # 进行采样
        sampled_data = sample_data(data, samples_needed)
        all_sampled_data.extend(sampled_data)
    
    # 打乱所有数据以确保随机性
    random.shuffle(all_sampled_data)
    
    # 写入输出文件
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in all_sampled_data:
            f.write(json.dumps(item) + '\n')

# 示例使用
tasks = ['position','color_attr', 'colors', 'counting', 'two_object']
weights = [0.7, 0.3, 0.1, 0.5, 0.1]  # 对应每个任务的权重，总和必须为1
sum_weights = sum(weights)
normalized_weights = [w / sum_weights for w in weights]
output_path = 'geneval_ood60_20/train_metadata.jsonl'

merge_datasets_with_weights(tasks, normalized_weights, output_path)