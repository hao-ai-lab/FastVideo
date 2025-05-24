import json
import os
from collections import defaultdict

import numpy as np


def configure_sta(mode='STA_searching', layer_num=40, time_step_num=50, head_num=40, **kwargs):
    """
    Configure Sliding Tile Attention (STA) parameters based on the specified mode.
    
    Parameters:
    ----------
    mode : str
        The STA mode to use. Options are:
        - 'STA_searching': Generate a set of mask candidates for initial search
        - 'STA_tuning': Select best mask strategy based on previously saved results
        - 'STA_inference': Load and use a previously tuned mask strategy
    layer_num: int, number of layers
    time_step_num: int, number of timesteps
    head_num: int, number of heads
    
    **kwargs : dict
        Mode-specific parameters:
        
        For 'STA_searching':
        - mask_candidates: list of str, optional, mask candidates to use
        - mask_selected: list of int, optional, indices of selected masks
        
        For 'STA_tuning':
        - mask_search_files_path: str, required, path to mask search results
        - mask_candidates: list of str, optional, mask candidates to use
        - mask_selected: list of int, optional, indices of selected masks
        - skip_time_steps: int, optional, number of time steps to use full attention (default 15)
        - save_dir: str, optional, directory to save mask strategy (default "mask_candidates")
        
        For 'STA_inference':
        - load_path: str, optional, path to load mask strategy (default "mask_candidates/mask_strategy.json")

    """
    valid_modes = ['STA_searching', 'STA_tuning', 'STA_inference', 'STA_tuning_cfg']
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}, got {mode}")

    if mode == 'STA_searching':
        # Get parameters with defaults
        mask_candidates = kwargs.get('mask_candidates')
        mask_selected = kwargs.get('mask_selected', list(range(len(mask_candidates))))

        # Parse selected masks
        selected_masks = []
        for index in mask_selected:
            mask = mask_candidates[index]
            masks_list = [int(x) for x in mask.split(',')]
            selected_masks.append(masks_list)

        # Create 3D mask structure with fixed dimensions (t=50, l=60)
        masks_3d = []
        for i in range(time_step_num):  # Fixed t dimension = 50
            row = []
            for j in range(layer_num):  # Fixed l dimension = 60
                row.append(selected_masks)  # Add all masks at each position
            masks_3d.append(row)

        return masks_3d

    elif mode == 'STA_tuning':
        # Get required parameters
        mask_search_files_path = kwargs.get('mask_search_files_path')
        if not mask_search_files_path:
            raise ValueError("mask_search_files_path is required for STA_tuning mode")

        # Get optional parameters with defaults
        mask_candidates = kwargs.get('mask_candidates')
        mask_selected = kwargs.get('mask_selected', list(range(len(mask_candidates))))
        skip_time_steps = kwargs.get('skip_time_steps')
        save_dir = kwargs.get('save_dir')

        # Parse selected masks
        selected_masks = []
        for index in mask_selected:
            mask = mask_candidates[index]
            masks_list = [int(x) for x in mask.split(',')]
            selected_masks.append(masks_list)

        # Read JSON results
        results = read_specific_json_files(mask_search_files_path)
        averaged_results = average_head_losses(results, selected_masks)

        # Add full attention mask for specific cases
        full_attention_mask = kwargs.get('full_attention_mask')
        selected_masks.append(full_attention_mask)

        # Select best mask strategy
        timesteps = kwargs.get('timesteps')
        mask_strategy, sparsity, strategy_counts = select_best_mask_strategy(averaged_results, selected_masks,
                                                                             skip_time_steps, timesteps, head_num)

        # Save mask strategy
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'mask_strategy_s{skip_time_steps}.json')
        with open(file_path, 'w') as f:
            json.dump(mask_strategy, f, indent=4)
        print(f"Successfully saved mask_strategy to {file_path}")

        # Print sparsity and strategy counts for information
        print(f"Overall sparsity: {sparsity:.4f}")
        print("\nStrategy usage counts:")
        total_heads = time_step_num * layer_num * head_num  # Fixed dimensions
        for strategy, count in strategy_counts.items():
            print(f"Strategy {strategy}: {count} heads ({count/total_heads*100:.2f}%)")

        # Convert dictionary to 3D list with fixed dimensions
        mask_strategy_3d = dict_to_3d_list(mask_strategy, t_max=time_step_num, l_max=layer_num, h_max=head_num)

        return mask_strategy_3d
    elif mode == 'STA_tuning_cfg':
        # Get required parameters for both positive and negative paths
        mask_search_files_path_pos  = kwargs.get('mask_search_files_path_pos')
        mask_search_files_path_neg = kwargs.get('mask_search_files_path_neg')
        save_dir = kwargs.get('save_dir')
        
        if not mask_search_files_path_pos or not mask_search_files_path_neg or not save_dir:
            raise ValueError("mask_search_files_path_pos, mask_search_files_path_neg, and save_dir are required for STA_tuning_cfg mode")

        # Get optional parameters with defaults
        mask_candidates = kwargs.get('mask_candidates')
        mask_selected = kwargs.get('mask_selected', list(range(len(mask_candidates))))
        skip_time_steps = kwargs.get('skip_time_steps')

        # Parse selected masks
        selected_masks = []
        for index in mask_selected:
            mask = mask_candidates[index]
            masks_list = [int(x) for x in mask.split(',')]
            selected_masks.append(masks_list)


        # Read JSON results for both positive and negative paths
        pos_results = read_specific_json_files(mask_search_files_path_pos)
        neg_results = read_specific_json_files(mask_search_files_path_neg)
        # from IPython import embed; embed()
        # Combine positive and negative results into one list
        combined_results = pos_results + neg_results
        
        # Average the combined results
        averaged_results = average_head_losses(combined_results, selected_masks)

        # Add full attention mask for specific cases
        full_attention_mask = kwargs.get('full_attention_mask')
        selected_masks.append(full_attention_mask)

        timesteps = kwargs.get('timesteps')
        # Select best mask strategy using combined results
        mask_strategy, sparsity, strategy_counts = select_best_mask_strategy(averaged_results, selected_masks,
                                                                             skip_time_steps, timesteps, head_num)

        # Save mask strategy
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, f'mask_strategy_s{skip_time_steps}.json')
        with open(file_path, 'w') as f:
            json.dump(mask_strategy, f, indent=4)
        print(f"Successfully saved mask_strategy to {file_path}")

        # Print sparsity and strategy counts for information
        print(f"Overall sparsity: {sparsity:.4f}")
        print("\nStrategy usage counts:")
        total_heads = time_step_num * layer_num * head_num  # Fixed dimensions
        for strategy, count in strategy_counts.items():
            print(f"Strategy {strategy}: {count} heads ({count/total_heads*100:.2f}%)")

        # Convert dictionary to 3D list with fixed dimensions
        mask_strategy_3d = dict_to_3d_list(mask_strategy, t_max=time_step_num, l_max=layer_num, h_max=head_num)

        return mask_strategy_3d

    else:  # STA_inference
        # Get parameters with defaults
        load_path = kwargs.get('load_path')

        # Load previously saved mask strategy
        with open(load_path, 'r') as f:
            mask_strategy = json.load(f)

        # Convert dictionary to 3D list with fixed dimensions
        mask_strategy_3d = dict_to_3d_list(mask_strategy, t_max=time_step_num, l_max=layer_num, h_max=head_num)

        return mask_strategy_3d


# Helper functions


def read_specific_json_files(folder_path):
    """Read and parse JSON files containing mask search results."""
    json_contents = []

    # List files only in the current directory (no walk)
    files = os.listdir(folder_path)
    # Filter files
    matching_files = [f for f in files if 'mask' in f and f.endswith('.json')]
    print(f"Found {len(matching_files)} matching files: {matching_files}")

    for file_name in matching_files:
        file_path = os.path.join(folder_path, file_name)
        with open(file_path, 'r') as file:
            data = json.load(file)
            json_contents.append(data)

    return json_contents


def average_head_losses(results, selected_masks):
    """Average losses across all prompts for each mask strategy."""
    # Initialize a dictionary to store the averaged results
    averaged_losses = {}
    loss_type = 'L2_loss'
    # Get all loss types (e.g., 'L2_loss')
    averaged_losses[loss_type] = {}

    for mask in selected_masks:
        mask_str = str(mask)
        data_shape = np.array(results[0][loss_type][mask_str]).shape
        accumulated_data = np.zeros(data_shape)

        # Sum across all prompts
        for prompt_result in results:
            accumulated_data += np.array(prompt_result[loss_type][mask_str])

        # Average by dividing by number of prompts
        averaged_data = accumulated_data / len(results)
        averaged_losses[loss_type][mask_str] = averaged_data

    return averaged_losses


def select_best_mask_strategy(averaged_results, selected_masks, skip_time_steps=15, timesteps=50, head_num=40):
    """Select the best mask strategy for each head based on loss minimization."""
    best_mask_strategy = {}
    loss_type = 'L2_loss'
    # Get the shape of time steps and layers
    time_steps = len(averaged_results[loss_type][str(selected_masks[0])])
    layers = len(averaged_results[loss_type][str(selected_masks[0])][0])

    # Counter for sparsity calculation
    total_tokens = 0  # total number of masked tokens
    total_length = 0  # total sequence length

    strategy_counts = {str(strategy): 0 for strategy in selected_masks}
    full_attn_strategy = selected_masks[-1]  # Last strategy is full attention
    print(f"Strategy {full_attn_strategy}, skip first {skip_time_steps} steps ")

    for t in range(timesteps):
        for l in range(layers):
            for h in range(head_num):
                if t < skip_time_steps:  # First steps use full attention
                    strategy = full_attn_strategy
                else:
                    # Get losses for this head across all strategies
                    head_losses = []
                    for strategy in selected_masks[:-1]:  # Exclude full attention
                        head_losses.append(averaged_results[loss_type][str(strategy)][t][l][h])

                    # Find which strategy gives minimum loss
                    best_strategy_idx = np.argmin(head_losses)
                    strategy = selected_masks[best_strategy_idx]

                best_mask_strategy[f'{t}_{l}_{h}'] = strategy

                # Calculate sparsity
                nums = strategy  # strategy is already a list of numbers
                total_tokens += nums[0] * nums[1] * nums[2]  # masked tokens for chosen strategy
                total_length += full_attn_strategy[0] * full_attn_strategy[1] * full_attn_strategy[2] 

                # Count strategy usage
                strategy_counts[str(strategy)] += 1

    overall_sparsity = 1 - total_tokens / total_length

    return best_mask_strategy, overall_sparsity, strategy_counts

# TODO: move to utils
def dict_to_3d_list(mask_strategy, t_max=50, l_max=60, h_max=24):
    result = [[[None for _ in range(h_max)] for _ in range(l_max)] for _ in range(t_max)]
    if mask_strategy is None:
        return result
    for key, value in mask_strategy.items():
        t, l, h = map(int, key.split('_'))
        result[t][l][h] = value
    return result

def save_mask_search_results(mask_search_final_result,
                             prompt,
                             mask_strategies,
                             output_dir='output/mask_search_result/'):
    if not mask_search_final_result:
        print("No mask search results to save")
        return None

    # Create result dictionary with defaultdict for nested lists
    mask_search_dict = {"L2_loss": defaultdict(list), "L1_loss": defaultdict(list)}

    mask_selected = list(range(len(mask_strategies)))
    selected_masks = []
    for index in mask_selected:
        mask = mask_strategies[index]
        masks_list = [int(x) for x in mask.split(',')]
        selected_masks.append(masks_list)

    # Process each mask strategy
    for i, mask_strategy in enumerate(selected_masks):
        mask_strategy = str(mask_strategy)
        # Process L2 loss
        step_results = []
        for step_data in mask_search_final_result:
            layer_losses = [layer_data["L2_loss"][i] for layer_data in step_data]
            step_results.append(layer_losses)
        mask_search_dict["L2_loss"][mask_strategy] = step_results

        step_results = []
        for step_data in mask_search_final_result:
            layer_losses = [layer_data["L1_loss"][i] for layer_data in step_data]
            step_results.append(layer_losses)
        mask_search_dict["L1_loss"][mask_strategy] = step_results

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename based on the first 20 characters of the prompt
    filename = prompt[:50].replace(" ", "_")
    filepath = os.path.join(output_dir, f'mask_search_{filename}.json')

    # Save the results to a JSON file
    with open(filepath, 'w') as f:
        json.dump(mask_search_dict, f, indent=4)

    print(f"Successfully saved mask research results to {filepath}")

    return filepath
