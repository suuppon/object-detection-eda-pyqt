import torch

def get_device(gpu_id=0):
    """
    Get the device for computation (Single GPU fallback).
    Returns 'cuda:id' if available, otherwise 'cpu'.
    For multi-gpu, typically DataParallel is handled internally by libraries.
    """
    if torch.cuda.is_available():
        if isinstance(gpu_id, int):
             if 0 <= gpu_id < torch.cuda.device_count():
                return f"cuda:{gpu_id}"
        elif isinstance(gpu_id, str):
             # If simple integer string
             if gpu_id.isdigit():
                 gid = int(gpu_id)
                 if 0 <= gid < torch.cuda.device_count():
                     return f"cuda:{gid}"
             # If multi-gpu string "0,1", just return default cuda:0 for explicit device selection context
             # Real multi-gpu handling is done via get_worker_device_arg for Ultralytics
             return "cuda:0"
        elif isinstance(gpu_id, list) and len(gpu_id) > 0:
            return f"cuda:{gpu_id[0]}"
            
    return "cpu"

def get_worker_device_arg(gpu_input):
    """
    Get the device argument format for Ultralytics/YOLO.
    
    Args:
        gpu_input: int (0), str ("0" or "0,1,2"), or list ([0, 1])
        
    Returns:
        int, list of ints, or 'cpu'
    """
    if not torch.cuda.is_available():
        return "cpu"

    device_count = torch.cuda.device_count()
    
    # 1. Handle Integer
    if isinstance(gpu_input, int):
        if 0 <= gpu_input < device_count:
            return gpu_input
        return "cpu"

    # 2. Handle List
    if isinstance(gpu_input, list):
        valid_ids = [g for g in gpu_input if isinstance(g, int) and 0 <= g < device_count]
        if valid_ids:
            return valid_ids if len(valid_ids) > 1 else valid_ids[0]
        return "cpu"

    # 3. Handle String ("0", "0,1", "0, 1")
    if isinstance(gpu_input, str):
        # Remove spaces
        clean_input = gpu_input.replace(" ", "")
        
        # If empty or 'cpu'
        if not clean_input or clean_input.lower() == 'cpu':
            return "cpu"
            
        parts = clean_input.split(',')
        valid_ids = []
        try:
            for p in parts:
                if p.isdigit():
                    gid = int(p)
                    if 0 <= gid < device_count:
                        valid_ids.append(gid)
        except ValueError:
            return "cpu"
            
        if valid_ids:
            return valid_ids if len(valid_ids) > 1 else valid_ids[0]
            
    return "cpu"
