import torch
import sys
import os

# Add the path to the built module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                             'ipc_handler_demo/cuda_tools'))

try:
    import ipc_tensor_pybind
except ImportError:
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

# Map PyTorch dtypes to scalar types (integers)
SCALAR_TYPE_MAP = {
    torch.float32: 0,    # kFloat
    torch.float64: 1,    # kDouble
    torch.float16: 2,    # kHalf
    torch.uint8: 3,      # kByte
    torch.int8: 4,       # kChar
    torch.int16: 5,      # kShort
    torch.int32: 6,      # kInt
    torch.int64: 7,      # kLong
    torch.bool: 11,      # kBool
    torch.bfloat16: 15,  # kBFloat16
}

def tensor_restore_from_handler_pybind(ipc_handle_bytes: bytes, meta):
    """
    Restore a tensor from an IPC handle using PyBind11.
    
    Args:
        ipc_handle_bytes: The IPC handle as bytes
        meta: Dictionary containing tensor metadata (shape, dtype, device)
    
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
    
    scalar_type = SCALAR_TYPE_MAP.get(dtype)
    if scalar_type is None:
        raise ValueError(f"No scalar type mapping for dtype: {dtype}")
    
    # Use PyBind11 module to restore tensor
    tensor = ipc_tensor_pybind.open_ipc_tensor(
        ipc_handle_bytes, 
        device, 
        list(shape), 
        dtype   
    )
    
    return tensor

def get_ipc_handle_pybind(tensor: torch.Tensor):
    """
    Get an IPC handle from a tensor using PyBind11.
    
    Args:
        tensor: The tensor to export
    
    Returns:
        tuple: (ipc_handle_bytes, meta) where meta contains tensor metadata
    """
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on a CUDA device")
    
    torch.cuda.synchronize(tensor.device)
    # Create metadata dictionary
    meta = {
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "device": tensor.device.index,
        "numel": tensor.numel(),
    }
    
    # Get IPC handle using PyBind11 module
    ipc_handle_bytes = ipc_tensor_pybind.export_ipc_handle(tensor)
    
    return ipc_handle_bytes, meta