# vllm/ipc_handler_demo/tensor_utils_pybind.py
import torch
import sys
import os

# Add the path to the built module
cuda_tools_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'ipc_handler_demo/cuda_tools')
sys.path.append(cuda_tools_path)

try:
    import ipc_tensor_pybind
except ImportError as e:
    print(f"Error importing ipc_tensor_pybind: {e}")
    print(f"Make sure you've built the module by running:")
    print(f"cd {cuda_tools_path} && pip install -e .")
    raise ImportError("Failed to import ipc_tensor_pybind. Make sure it's built and in the Python path.")

# Map string dtype names to torch.dtype objects
DTYPE_MAP = {
    'torch.float32': torch.float32,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'torch.float64': torch.float64,
    'torch.uint8': torch.uint8,
    'torch.float16': torch.float16,
    'torch.bfloat16': torch.bfloat16,
}

def tensor_restore_from_handler_pybind(ipc_handle_bytes: bytes, meta, make_contiguous=False):
    """
    Restore a tensor from an IPC handle using PyBind11.
    
    Args:
        ipc_handle_bytes: The IPC handle as bytes
        meta: Dictionary containing tensor metadata (shape, dtype, device)
        make_contiguous: Whether to make the tensor contiguous (creates a copy)
    
    Returns:
        torch.Tensor: The restored tensor
    """
    shape = meta['shape']
    dtype_str = meta['dtype']
    device = meta['device']
    
    if len(ipc_handle_bytes) != 64:
        raise ValueError("Invalid IPC handle size")

    # Convert string dtype to torch dtype
    dtype = DTYPE_MAP.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    
    # Ensure the device is ready
    torch.cuda.synchronize(device)
    
    # Use PyBind11 module to restore tensor
    tensor = ipc_tensor_pybind.open_ipc_tensor(
        ipc_handle_bytes, 
        device, 
        list(shape), 
        dtype,
        make_contiguous
    )
    
    # If strides were provided in metadata, try to respect them
    if 'strides' in meta and not make_contiguous:
        # This is a best-effort approach, as reshaping with strides is complex
        # For now, we'll just verify if the tensor is contiguous when it should be
        is_contiguous_in_meta = meta.get('is_contiguous', True)
        if is_contiguous_in_meta != tensor.is_contiguous():
            print(f"Warning: Tensor contiguity mismatch. Expected: {is_contiguous_in_meta}, Got: {tensor.is_contiguous()}")
    
    return tensor

def get_ipc_handle_pybind(tensor: torch.Tensor, ensure_contiguous=True):
    """
    Get an IPC handle from a tensor using PyBind11.
    
    Args:
        tensor: The tensor to export
        ensure_contiguous: Whether to ensure the tensor is contiguous before exporting
    
    Returns:
        tuple: (ipc_handle_bytes, meta) where meta contains tensor metadata
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device")
    
    # Synchronize to ensure all operations on the tensor are complete
    torch.cuda.synchronize(tensor.device)
    
    # Create metadata dictionary with more detailed information
    meta = {
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "device": tensor.device.index,
        "numel": tensor.numel(),
        "strides": tensor.stride(),
        "is_contiguous": tensor.is_contiguous(),
    }
    
    # Get IPC handle using PyBind11 module
    ipc_handle_bytes = ipc_tensor_pybind.export_ipc_handle(tensor, ensure_contiguous)
    
    return ipc_handle_bytes, meta