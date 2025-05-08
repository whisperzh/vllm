#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>
#include <filesystem>


namespace fs = std::filesystem;

int main() {
    std::string directory_path = "/home/ubuntu/vllm_test_field/vllm/demo/weights";
    std::string handle_path = "/home/ubuntu/vllm_test_field/vllm/demo/handles";


    for(int i=0;i<4;i++)
    {
        cudaSetDevice(i);
        std::string device_path = handle_path + "/device_" + std::to_string(i);

        fs::create_directories(device_path);
        for (const auto& entry : fs::directory_iterator(directory_path)) {
            const auto& path = entry.path();
            std::string filename = path.filename().string();
            if (filename.find("model_layers_23_mlp") == std::string::npos) {
                continue; // 只处理 .bin 文件
            }
            std::streampos file_size;
            std::ifstream file(path, std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Failed to open file " << filename << std::endl;
                continue;
            }
    
            file.seekg(0, std::ios::end);
            file_size = file.tellg();
            file.seekg(0, std::ios::beg);
            size_t num_elements = file_size / sizeof(float);
            std::vector<float> host_data(num_elements);
            file.read(reinterpret_cast<char*>(host_data.data()), num_elements * sizeof(float));
            file.close();
    
            float* device_data;
            cudaMalloc(&device_data, num_elements * sizeof(float));
            cudaMemcpy(device_data, host_data.data(), num_elements * sizeof(float), cudaMemcpyHostToDevice);
    
            // 获取 IPC handle
            cudaIpcMemHandle_t handle;
            cudaIpcGetMemHandle(&handle, device_data);
    
            // 保存 handle
            std::string handle_filename = device_path + "/" + filename + ".ipc";
            std::ofstream handle_file(handle_filename, std::ios::binary);
            handle_file.write(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t));
            handle_file.close();
    
            std::cout << "Handle saved: " << handle_filename << std::endl;
    
            
        }
    
      
    }
    // 确保 handle 目录存在

  // Keep the program running
        printf("Data initialized. Press Enter to exit...\n");
        getchar();


    return 0;
}