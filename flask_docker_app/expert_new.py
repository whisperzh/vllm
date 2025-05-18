import torch
import os
import json
from flask import Flask, request, jsonify
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl 
from tensor_utils import get_ipc_handle, tensor_restore_from_handler, tensor_restore_from_handler2
### This is a simple Flask app that loads expert weights and performs forward pass without ipc handler
app = Flask(__name__)

RANK = int(os.environ.get("RANK", 0))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 30))
LOADED_EXPERTS = list(os.environ.get("LOADED_EXPERTS", range(RANK*NUM_EXPERTS,(RANK+1)*NUM_EXPERTS))) 

GPU_IDX = int(os.environ.get("GPU_IDX", RANK))
WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "/root/vllm_test/vllm/ipc_handler_demo/weights")
LAYER = int(os.environ.get("LAYER", 0))
GLOBAL_NUM_EXPERTS = int(os.environ.get("GLOBAL_NUM_EXPERTS", 60))
SELF_WARMUP = bool(os.environ.get("SELF_WARMUP", False))
cuda_device = f"cuda:{GPU_IDX}"


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
        
        weight1_path=os.path.join(path, f"{prefix}{expert}_gate_proj_weight.pt")
        weight3_path=os.path.join(path, f"{prefix}{expert}_up_proj_weight.pt")
        w13=torch.cat([torch.load(weight1_path).to(device=cuda_device), torch.load(weight3_path).to(device=cuda_device)], dim=0)
        w1.append(w13)
        
        weight2_path=os.path.join(path, f"{prefix}{expert}_down_proj_weight.pt")
        w2.append(torch.load(weight2_path).to(device=cuda_device))
        
    w1,w2=torch.stack(w1,dim=0).to(cuda_device),  torch.stack(w2,dim=0).to(cuda_device)
    return w1,w2

expert_map_generated=init_expert_map(GLOBAL_NUM_EXPERTS)
w1,w2 = load_expert_weights(WEIGHT_PATH,LAYER)
print(f"Loaded experts' index: {LOADED_EXPERTS},\n")

def moe_forward(worker, inputs):
    
    output = fused_experts_impl(
        hidden_states = inputs["hidden_states"],
        w1 = worker["w1"],
        w2 = worker["w2"],
        topk_weights = inputs["topk_weights"],
        topk_ids = inputs["topk_ids"],
        inplace = True,
        activation = "silu",
        expert_map = inputs["expert_map"],
        global_num_experts = GLOBAL_NUM_EXPERTS 
    )
    return output

if(SELF_WARMUP):
    print("Warmup the model")
    hidden_states = torch.randn(128, 2048, dtype=torch.bfloat16).to(cuda_device)
    topk_weights = torch.randn(128, 4, dtype=torch.float32).to(cuda_device)
    topk_ids = torch.randint(0, 60, (128, 4), dtype=torch.int32).to(cuda_device)
    inputs={
        "hidden_states": hidden_states,
        "topk_weights": topk_weights,
        "topk_ids": topk_ids,
        "expert_map": expert_map_generated
        
    }
    worker = {
        "w1": w1,
        "w2": w2
    }
    _ = moe_forward(worker, inputs)
    print("Warmup done")



@app.route("/forward", methods=["POST"])
def forward():

    hidden_states_handler = request.files['hidden_states_handler'].read()
    topk_weights_handler = request.files['topk_weights_handler'].read()
    topk_ids_handler = request.files['topk_ids_handler'].read()
    
    # 2. 获取字典数据（JSON 格式）
    hidden_states_meta = json.loads(request.form['hidden_states_meta']) 
    topk_weights_meta = json.loads(request.form['topk_weights_meta']) 
    topk_ids_meta = json.loads(request.form['topk_ids_meta']) 
    
    hidden_states = tensor_restore_from_handler2(hidden_states_handler, hidden_states_meta)
    topk_weights = tensor_restore_from_handler2(topk_weights_handler, topk_weights_meta)
    topk_ids = tensor_restore_from_handler2(topk_ids_handler, topk_ids_meta)
    
    expert_map = torch.tensor(request.form["expert_map"], dtype=torch.int32,device=cuda_device) if request.form.get("expert_map",None) is not None else expert_map_generated

    inputs={
        "hidden_states": torch.tensor(hidden_states, dtype=torch.bfloat16,device=cuda_device),
        "topk_weights": torch.tensor(topk_weights, dtype=torch.float32,device=cuda_device),
        "topk_ids": torch.tensor(topk_ids, dtype=torch.int32,device=cuda_device),
        "expert_map": expert_map
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
