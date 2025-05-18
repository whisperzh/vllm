import cupy as cp
import torch
import ctypes

lib_open_handler = ctypes.CDLL('/root/vllm_test/vllm/ipc_handler_demo/ipc_handle.so')
lib_open_handler.open_ipc_handle.argtypes = [ctypes.c_void_p]
lib_open_handler.open_ipc_handle.restype = ctypes.c_void_p

lib_get_handler = ctypes.CDLL('/root/vllm_test/vllm/ipc_handler_demo/cuda_tools/libipc_tensor_tool.so')
lib_get_handler.export_ipc_handle.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
lib_get_handler.export_ipc_handle.restype = ctypes.c_int

def tensor_restore_from_handler(handler_bytes, meta):
    torch.cuda.set_device(int(meta['device']))
    device_ptr = lib_open_handler.open_ipc_handle(handler_bytes)
    if device_ptr:
        tensor_size = meta['numel']
        dtype_map = {
            'torch.float32': cp.float32,
            'torch.float64': cp.float64,
            'torch.int32': cp.int32,
            'torch.int64': cp.int64,
            'torch.uint8': cp.uint8,
            'torch.int8': cp.int8,
            'torch.int16': cp.int16,
            'torch.float16': cp.float16,
            'torch.bfloat16': cp.float16  # Map bfloat16 to float16 since CuPy doesn't support bfloat16
        }
        # 使用提前知道的dtype信息
        cp_dtype = dtype_map.get(meta['dtype'])
        if cp_dtype is None:
            raise ValueError(f"Unsupported dtype: {meta['dtype']}")

        unownedmemory = cp.cuda.UnownedMemory(
            device_ptr, tensor_size * cp_dtype().itemsize, None)
        # Wrap the raw GPU pointer using CuPy
        gpu_array = cp.ndarray((tensor_size,), dtype=cp_dtype, memptr=cp.cuda.MemoryPointer(unownedmemory, 0))

        # Convert CuPy array to PyTorch tensor using DLPack
        dlpack = gpu_array.toDlpack()
        restored = torch.utils.dlpack.from_dlpack(dlpack).view(meta['shape'])
        
        # If original dtype was bfloat16, convert back to bfloat16
        if meta['dtype'] == 'torch.bfloat16':
            restored = restored.to(torch.bfloat16)
            
        return restored
    else:
        print(f"Failed to open IPC handle")
        return None
    
def get_ipc_handle(tensor: torch.Tensor) -> bytes:
    
    meta={
        "shape": tensor.shape,
        "dtype": str(tensor.dtype),
        "device": int(tensor.device.index),
        "numel": tensor.numel(),
    }
    if not tensor.is_cuda:
        raise ValueError("Tensor must be on CUDA device")

    dev_ptr = tensor.data_ptr()
    out = ctypes.create_string_buffer(64)

    result = lib_get_handler.export_ipc_handle(ctypes.c_void_p(dev_ptr), out)
    if result != 0:
        raise RuntimeError(f"export_ipc_handle failed with code {result}")

    return out.raw, meta 


# Load shared library
lib3 = ctypes.CDLL('/root/vllm_test/vllm/ipc_handler_demo/cuda_tools/libipc_tensor_tool.so')
lib3.open_ipc_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib3.open_ipc_tensor.restype = ctypes.c_void_p

lib3.close_ipc_tensor.argtypes = [ctypes.c_void_p]
lib3.close_ipc_tensor.restype = ctypes.c_int


DTYPE_SIZE = {
    torch.float32: 4,
    torch.int32: 4,
    torch.int64: 8,
    torch.float64: 8,
    torch.uint8: 1,
    torch.float16: 2,
    torch.bfloat16: 2,
    # add more if needed
}

DTYPE_MAP = {
    'torch.float32': torch.float32,
    'torch.int32': torch.int32,
    'torch.int64': torch.int64,
    'torch.float64': torch.float64,
    'torch.uint8': torch.uint8,
    'torch.float16': torch.float16,
    'torch.bfloat16': torch.bfloat16,
}

def tensor_restore_from_handler2(ipc_handle_bytes: bytes, meta):
    shape = meta['shape']
    dtype = meta['dtype']
    device = meta['device']
    
    if len(ipc_handle_bytes) != 64:
        raise ValueError("Invalid IPC handle size")

    dtype = DTYPE_MAP.get(dtype, None)
    if dtype not in DTYPE_SIZE:
        raise ValueError(f"Unsupported dtype: {dtype}")

    handle_buf = ctypes.create_string_buffer(ipc_handle_bytes, 64)
    dev_ptr = lib3.open_ipc_tensor(handle_buf, device)

    if not dev_ptr:
        raise RuntimeError("Failed to open IPC handle")

    numel = torch.prod(torch.tensor(shape)).item()
    nbytes = numel * DTYPE_SIZE[dtype]

    # Create a buffer from the device pointer
    buffer = (ctypes.c_char * nbytes).from_address(dev_ptr)
    
    # Special handling for bfloat16
    if dtype == torch.bfloat16:
        # First create as uint16 then convert to bfloat16
        t = torch.frombuffer(buffer, dtype=torch.uint16).view(*shape).to(f'cuda:{device}')
        t = t.view(torch.bfloat16)
    else:
        # Normal path for other dtypes
        t = torch.frombuffer(buffer, dtype=dtype).view(*shape).to(f'cuda:{device}')
    
    # Close the IPC handle
    lib3.close_ipc_tensor(dev_ptr)
    return t