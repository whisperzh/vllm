docker build -t  moe_expert .

docker run -d \
  --name fused_moe_layer_23_exp_0_29 \
  --gpus all \
  -p 5001:5000 \
  -v /home/ubuntu/vllm_test_field/vllm/demo/weights:/app/weights \
  -e "RANK=0" \
  -e "LOADED_EXPERTS=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,20,21,22,23,24,25,26,27,28,29]" \
  -e "GPU_IDX=0" \
  -e "WEIGHT_PATH=/app/weights" \
  -e "LAYER=23" \
  moe_expert
  

  docker run -d \
  --name fused_moe_layer_23_exp30_59 \
  --gpus all \
  -p 5001:5000 \
  -v /home/ubuntu/vllm_test_field/vllm/demo/weights:/app/weights \
  -e "RANK=0" \
  -e "LOADED_EXPERTS=[30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]" \
  -e "GPU_IDX=0" \
  -e "WEIGHT_PATH=/app/weights" \
  -e "LAYER=23" \
  moe_expert