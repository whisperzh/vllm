import torch
import sys
import os

# Add the path to the flask_docker_app directory
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'flask_docker_app'))

# Import the PyBind11-based functions
from tensor_utils_pybind import get_ipc_handle_pybind, tensor_restore_from_handler_pybind

def test_ipc_tensor_pybind():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Skipping test.")
        return
    
    # Create a test tensor on GPU
    device = torch.device('cuda:0')
    original_tensor = torch.randn(3, 4, 5, device=device)
    print(f"Original tensor: {original_tensor}")
    
    # Get IPC handle
    ipc_handle_bytes, meta = get_ipc_handle_pybind(original_tensor)
    print(f"Got IPC handle, size: {len(ipc_handle_bytes)} bytes")
    print(f"Metadata: {meta}")
    
    # Restore tensor from IPC handle
    restored_tensor = tensor_restore_from_handler_pybind(ipc_handle_bytes, meta)
    print(f"Restored tensor: {restored_tensor}")
    
    # Check if tensors match
    is_equal = torch.allclose(original_tensor, restored_tensor)
    print(f"Tensors match: {is_equal}")
    
    # Test with different dtypes
    dtypes = [torch.float32, torch.int32, torch.float16]
    for dtype in dtypes:
        print(f"\nTesting with dtype: {dtype}")
        original = torch.randint(0, 100, (2, 3), dtype=dtype, device=device)
        if dtype == torch.float16 or dtype == torch.float32:
            original = torch.randn(2, 3, dtype=dtype, device=device)
        
        print(f"Original: {original}")
        
        # Export and restore
        ipc_handle, meta = get_ipc_handle_pybind(original)
        restored = tensor_restore_from_handler_pybind(ipc_handle, meta)
        
        print(f"Restored: {restored}")
        is_equal = torch.all(original == restored)
        print(f"Equal: {is_equal}")

if __name__ == "__main__":
    test_ipc_tensor_pybind()