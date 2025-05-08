from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts_impl 

app = Flask(__name__)

RANK = int(os.environ.get("RANK", 0))
LOADED_EXPERTS = list(os.environ.get("EXPERTS", [0,1,2,3,4,5,6,7,8,9]))
GPU_IDX = int(os.environ.get("RANK", 0))
WEIGHT_PATH = os.environ.get("WEIGHT_PATH", "/app/weights")

def init_expert_map(total_expert_num):
    expert_map = [-1]*total_expert_num
    for e in LOADED_EXPERTS:
        expert_map[e] = e
    return torch.tensor(expert_map, device="cuda", dtype=torch.int32)

expert_map=init_expert_map(60)



def load_expert_weights(path, layer):
    w1 = []
    w2 = []
    prefix=f"model_layers_{layer}_mlp_experts_"
    for expert in LOADED_EXPERTS:
        
        weight1_path=os.path.join(path, f"{prefix}{expert}_gate_proj_weight.bin")
        weight3_path=os.path.join(path, f"{prefix}{expert}_up_proj_weight.bin")
        w13=torch.cat([torch.load(weight1_path).to(device="cuda"), torch.load(weight3_path).to(device="cuda")], dim=0)
        w1.append(w13)
        
        weight2_path=os.path.join(path, f"{prefix}{expert}_down_proj_weight.bin")
        w2.append(torch.load(weight2_path).to(device="cuda"))
        
    w1,w2=torch.stack(w1,dim=0).to("cuda"),  torch.stack(w2,dim=0).to("cuda")
    return w1,w2

w1,w2 = load_expert_weights(WEIGHT_PATH,0)

@app.route("/forward", methods=["POST"])
def forward():
    hidden_states = request.json["hidden_states"]
    topk_weights = request.json["topk_weights"]
    topk_ids = request.json["topk_ids"]

    inputs={
        "hidden_states": torch.tensor(hidden_states,dtype=torch.bfloat16).to("cuda"),
        "topk_weights": torch.tensor(topk_weights, dtype=torch.float32).to("cuda"),
        "topk_ids": torch.tensor(topk_ids, dtype=torch.int32).to("cuda")
    }
    worker = {
        "w1": w1,
        "w2": w2
    }

    output_tensor = moe_forward(worker, inputs).cpu()
    return jsonify({"hidden_output": output_tensor.tolist()})

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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
