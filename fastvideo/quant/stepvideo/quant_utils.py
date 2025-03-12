import torch
import gc

from quant_layer import FP8Linear

def get_model_size(model: torch.nn.Module):
    """
    Computes the memory footprint of a PyTorch model in megabytes (MB).
    """
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    total_buffers = sum(b.numel() * b.element_size() for b in model.buffers())
    total_size = (total_params + total_buffers) / (1024 ** 2)  # Convert bytes to MB
    return total_size

def check_module_device_consistency(module: torch.nn.Module):
    devices = {param.device for param in module.parameters()}
    return len(devices) == 1

def quant_layer_refactor_(submodule,name,parent_module,quant_config,full_name):
    input_size = submodule.in_features
    output_size = submodule.out_features
    bias = True if submodule.bias is not None else False
    skip_bias_add = False if submodule.bias is not None else True
    params_dtype = submodule.weight.dtype
    device = submodule.weight.device
    prefix = full_name

    quant_linear = FP8Linear(input_size, output_size, bias=bias, skip_bias_add=skip_bias_add, params_dtype=params_dtype, quant_config=quant_config, prefix=prefix)
    quant_linear.weight.copy_(submodule.weight)
    del submodule
    
    quant_linear = quant_linear.to(device)
    quant_linear.quant_method.process_weights_after_loading(quant_linear)
    
    setattr(parent_module, name, quant_linear)
    gc.collect()
    torch.cuda.empty_cache()
        
def apply_func_to_submodules(module, class_type, function, parent_name="", quant_layers=[], **kwargs):
    """
    Recursively iterates through all submodules of a PyTorch module and applies a hook function
    if the submodule matches the specified class type. The parent name is appended to the submodule name.

    Args:
        module (torch.nn.Module): The PyTorch module to iterate through.
        class_type (type): The class type to match against submodules.
        function (callable): The function to apply if a submodule matches the class type.
        parent_name (str): The name of the parent module (used for recursion).
    """

    for name, submodule in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name
        parent_module = module

        # INFO: pass from the parent call into func
        if 'name' in kwargs:
            kwargs['name']=name
        if 'full_name' in kwargs:
            kwargs['full_name'] = full_name
        if 'parent_module' in kwargs:
            kwargs['parent_module'] = module
        if isinstance(submodule, class_type):
            for quant_layer in quant_layers:
                if quant_layer in full_name:
                    function(submodule, **kwargs)
                    break

        # Recursively apply the function to submodules
        apply_func_to_submodules(submodule, class_type, function, parent_name=full_name, quant_layers=quant_layers, **kwargs)