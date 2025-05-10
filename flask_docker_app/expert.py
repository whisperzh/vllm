from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl 

app = Flask(__name__)

RANK = int(os.environ.get("RANK", 0))
NUM_EXPERTS = int(os.environ.get("NUM_EXPERTS", 30))
LOADED_EXPERTS = list(range(RANK*NUM_EXPERTS,(RANK+1)*NUM_EXPERTS)) 

GPU_IDX = int(os.environ.get("GPU_IDX", RANK))
WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "/home/ubuntu/vllm_test_field/vllm/ipc_handler_demo/weights")
LAYER = int(os.environ.get("LAYER", 0))
GLOBAL_NUM_EXPERTS = int(os.environ.get("GLOBAL_NUM_EXPERTS", 60))

cuda_device = f"cuda:{GPU_IDX}"


def init_expert_map(total_expert_num):
    print(f"LOADED_EXPERTS: {LOADED_EXPERTS}")
    expert_map = [-1]*total_expert_num
    for idx,e in enumerate(LOADED_EXPERTS):
        expert_map[e] = idx
    return torch.tensor(expert_map, device=cuda_device, dtype=torch.int32)

def load_expert_weights(path, layer):
    print(f"LOADED_EXPERTS: {LOADED_EXPERTS}")
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
        expert_map = expert_map,
        global_num_experts = 60 
    )
    return output



@app.route("/forward", methods=["POST"])
def forward():
    hidden_states = request.json["hidden_states"]
    topk_weights = request.json["topk_weights"]
    topk_ids = request.json["topk_ids"]

    inputs={
        "hidden_states": torch.tensor(hidden_states, dtype=torch.bfloat16,device=cuda_device),
        "topk_weights": torch.tensor(topk_weights, dtype=torch.float32,device=cuda_device),
        "topk_ids": torch.tensor(topk_ids, dtype=torch.int32,device=cuda_device)
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
