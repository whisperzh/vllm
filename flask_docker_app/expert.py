import torch
import os
import ctypes
import json
import torch.nn as nn
from flask import Flask, request, jsonify
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl 

app = Flask(__name__)

RANK = int(os.environ.get("RANK", 0))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 30))
LOADED_EXPERTS = list(os.environ.get("LOADED_EXPERTS", range(RANK*NUM_EXPERTS,(RANK+1)*NUM_EXPERTS))) 

GPU_IDX = int(os.environ.get("GPU_IDX", RANK))
WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "/home/ubuntu/vllm_test_field/vllm/ipc_handler_demo/weights")
LAYER = int(os.environ.get("LAYER", 0))
GLOBAL_NUM_EXPERTS = int(os.environ.get("GLOBAL_NUM_EXPERTS", 60))

cuda_device = f"cuda:{GPU_IDX}"



# Load the shared library
lib = ctypes.CDLL('/home/ubuntu/vllm_test_field/vllm/flask_docker_app/cuda_tools/libipc_tensor_tool.so')
lib.open_ipc_tensor.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.open_ipc_tensor.restype = ctypes.c_void_p

lib.close_ipc_tensor.argtypes = [ctypes.c_void_p]
lib.close_ipc_tensor.restype = ctypes.c_int


DTYPE_SIZE = {
    torch.float32: 4,
    torch.int32: 4,
    torch.int64: 8,
    torch.float64: 8,
    torch.uint8: 1,
    torch.bfloat16: 2,
    # add more if needed
}

DTYPE_MAP = {
    'torch.float32':torch.float32,
    'torch.int32':torch.int32,
    'torch.int64':torch.int64,
    'torch.float64':torch.float64,
    'torch.uint8':torch.uint8,
    'torch.float16':torch.float16,
    'torch.bfloat16':torch.bfloat16,
}

def restore_tensor(ipc_handle_bytes: bytes, shape, dtype=torch.float32, device=0):
    if len(ipc_handle_bytes) != 64:
        raise ValueError("Invalid IPC handle size")

    dtype = DTYPE_MAP.get(dtype, None)
    if dtype not in DTYPE_SIZE:
        raise ValueError(f"Unsupported dtype: {dtype}")

    handle_buf = ctypes.create_string_buffer(ipc_handle_bytes, 64)
    dev_ptr = lib.open_ipc_tensor(handle_buf, device)

    if not dev_ptr:
        raise RuntimeError("Failed to open IPC handle")

    numel = torch.prod(torch.tensor(shape)).item()
    nbytes = numel * DTYPE_SIZE[dtype]

    # Wrap the pointer as a ctypes pointer of the right type
    # (important: we cast to ctypes type matching dtype)
    ptr_type = ctypes.POINTER(ctypes.c_float)  # default
    if dtype == torch.float32:
        ptr_type = ctypes.POINTER(ctypes.c_float)
    elif dtype == torch.int32:
        ptr_type = ctypes.POINTER(ctypes.c_int32)
    elif dtype == torch.int64:
        ptr_type = ctypes.POINTER(ctypes.c_int64)
    elif dtype == torch.float64:
        ptr_type = ctypes.POINTER(ctypes.c_double)
    elif dtype == torch.uint8:
        ptr_type = ctypes.POINTER(ctypes.c_uint8)
    elif dtype == torch.bfloat16:
        ptr_type = ctypes.POINTER(ctypes.c_uint16)
    else:
        raise ValueError(f"Unsupported dtype for ctypes cast: {dtype}")

    typed_ptr = ctypes.cast(dev_ptr, ptr_type)

    # Use torch.from_blob (no ownership)
    t = torch.frombuffer(
        (ctypes.c_char * nbytes).from_address(dev_ptr),
        dtype=dtype
    ).view(*shape).to(f'cuda:{device}')
    lib.close_ipc_tensor(dev_ptr)
    return t

# End loading shared library


def init_expert_map(total_expert_num):
    expert_map = [-1]*total_expert_num
    for idx,e in enumerate(LOADED_EXPERTS):
        expert_map[e] = idx
    return torch.tensor(expert_map, device=cuda_device, dtype=torch.int32)

def load_expert_weights(path, layer):
    w1 = []
    w2 = []
    prefix=f"model_layers_{layer}_mlp_experts_"
    for expert in LOADED_EXPERTS:
        
        weight1_path=os.path.join(path, f"{prefix}{expert}_gate_proj_weight.bin")
        weight3_path=os.path.join(path, f"{prefix}{expert}_up_proj_weight.bin")
        w13=torch.cat([torch.load(weight1_path).to(device=cuda_device), torch.load(weight3_path).to(device=cuda_device)], dim=0)
        w1.append(w13)
        
        weight2_path=os.path.join(path, f"{prefix}{expert}_down_proj_weight.bin")
        w2.append(torch.load(weight2_path).to(device=cuda_device))
        
    w1,w2=torch.stack(w1,dim=0).to(cuda_device),  torch.stack(w2,dim=0).to(cuda_device)
    return w1,w2

expert_map=init_expert_map(GLOBAL_NUM_EXPERTS)
w1,w2 = load_expert_weights(WEIGHT_PATH,LAYER)

def moe_forward(worker, inputs):
    
    output = fused_experts_impl(
        hidden_states = inputs["hidden_states"],
        w1 = worker["w1"],
        w2 = worker["w2"],
        topk_weights = inputs["topk_weights"],
        topk_ids = inputs["topk_ids"],
        inplace = True,
        activation = "silu",
        expert_map = expert_map if inputs.get("expert_map",None) is None else inputs["expert_map"],
        global_num_experts = GLOBAL_NUM_EXPERTS 
    )
    return output



@app.route("/forward", methods=["POST"])
def forward():
    # hidden_states = request.json["hidden_states_handler"]
    # topk_weights = request.json["topk_weights_handler"]
    # topk_ids = request.json["topk_ids_handler"]
    
    hidden_states_handler = request.files['hidden_states_handler'].read()
    topk_weights_handler = request.files['topk_weights_handler'].read()
    topk_ids_handler = request.files['topk_ids_handler'].read()
    
    # 2. 获取字典数据（JSON 格式）
    hidden_states_meta = json.loads(request.form['hidden_states_meta']) 
    topk_weights_meta = json.loads(request.form['topk_weights_meta']) 
    topk_ids_meta = json.loads(request.form['topk_ids_meta']) 
    

    inputs={
        "hidden_states": restore_tensor(hidden_states_handler, hidden_states_meta["shape"], hidden_states_meta["dtype"],  hidden_states_meta["device"]),
        "topk_weights": restore_tensor(topk_weights_handler, topk_weights_meta["shape"], topk_weights_meta["dtype"],  topk_weights_meta["device"]),
        "topk_ids": restore_tensor(topk_ids_handler, topk_ids_meta["shape"], topk_ids_meta["dtype"],  topk_ids_meta["device"]),
    }
    worker = {
        "w1": w1,
        "w2": w2
    }
    torch.cuda.set_device(GPU_IDX)
    print(f"Environment Variables: RANK={RANK},\n\
            LOADED_EXPERTS={LOADED_EXPERTS},\n\
            GPU_IDX={GPU_IDX},\n\
            WEIGHT_PATH={WEIGHT_PATH}, \n\
            LAYER={LAYER},\n\
            GLOBAL_NUM_EXPERTS={GLOBAL_NUM_EXPERTS},\n\
            On the same device {w1.device==w2.device==inputs['hidden_states'].device==expert_map.device},\n\
            current cuda device: {torch.cuda.current_device()},\n\
            current var device: {w1.device},\n\
            expert_map: {expert_map}")
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    output_tensor = moe_forward(worker, inputs)
    end_event.record()
    torch.cuda.synchronize()
    latency_ms = start_event.elapsed_time(end_event)
    return jsonify({"hidden_output": output_tensor.cpu().tolist(),"latency_ms":latency_ms})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
