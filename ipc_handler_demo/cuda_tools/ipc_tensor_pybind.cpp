    // vllm/ipc_handler_demo/cuda_tools/ipc_tensor_pybind.cpp
    #include <pybind11/pybind11.h>
    #include <pybind11/numpy.h>
    #include <pybind11/stl.h>
    #include <cuda_runtime.h>
    #include <torch/extension.h>
    #include <string>
    #include <vector>

    namespace py = pybind11;

    void* open_ipc_handle_pybind(py::bytes handle_bytes, int device) {
        // Get the size of the bytes object using PyBytes_Size
        Py_ssize_t size = PyBytes_Size(handle_bytes.ptr());
        if (size != sizeof(cudaIpcMemHandle_t)) {
            throw std::runtime_error("Invalid IPC handle size");
        }

        // Set CUDA device
        cudaError_t err = cudaSetDevice(device);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaSetDevice failed: ") + cudaGetErrorString(err));
        }

        // Copy handle from bytes
        cudaIpcMemHandle_t handle;
        std::memcpy(&handle, PyBytes_AsString(handle_bytes.ptr()), sizeof(cudaIpcMemHandle_t));

        void* dev_ptr = nullptr;
        cudaError_t err1 = cudaDeviceSynchronize();
        if (err1 != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaIpcOpenMemHandle failed: ") + cudaGetErrorString(err1));
        }


        err = cudaIpcOpenMemHandle(&dev_ptr, handle, cudaIpcMemLazyEnablePeerAccess);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaIpcOpenMemHandle failed: ") + cudaGetErrorString(err));
        }

        return dev_ptr;
    }

    // 新增：从已打开的handle创建tensor的函数
    torch::Tensor create_tensor_from_ptr(void* dev_ptr,
                                    size_t offset_bytes,
                                    std::vector<int64_t> shape,
                                    torch::Dtype dtype,
                                    int device,
                                    bool make_contiguous) {
        void* offset_ptr = static_cast<void*>(static_cast<char*>(dev_ptr) + offset_bytes);
        
        std::cout << "offset_ptr: " << offset_ptr << std::endl;
        // Synchronize CUDA device to ensure memory is available
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
        }


        // // Create PyTorch tensor from device pointer
        // auto options = torch::TensorOptions()
        //     .dtype(dtype)
        //     .device(torch::kCUDA, device);
        
        // // Create tensor without copying data
        // auto tensor = torch::from_blob(offset_ptr, shape, 
        //     // Empty deleter since we'll handle IPC handle closing separately
        //     [](void*) {}, 
        //     options);



        auto options = torch::TensorOptions().dtype(dtype).
        device(torch::kCUDA, device);

        // Create an empty tensor with the specified shape and dtype
        auto tensor = torch::empty(shape, options);

        // Copy the data from the device pointer to the newly allocated tensor
        cudaMemcpy(tensor.data_ptr(), offset_ptr, tensor.nbytes(), 
        cudaMemcpyDeviceToDevice);




        std::cout << "tensor.data_ptr: " << tensor.data_ptr() << std::endl;
        std::cout << "Last 5 elements of the restored tensor " 
              << tensor.slice(0, tensor.size(0) - 5) << std::endl;
        // Return the tensor, either as is or as a clone
        if (make_contiguous) {
            return tensor.clone();
        } else {
            return tensor;
        }

    }

    // Function to export an IPC handle from a PyTorch tensor
    py::bytes export_ipc_handle_pybind(torch::Tensor tensor, bool ensure_contiguous) {
        if (!tensor.is_cuda()) {
            throw std::runtime_error("Tensor must be on a CUDA device");
        }
        
        // Make tensor contiguous if requested
        if (ensure_contiguous && !tensor.is_contiguous()) {
            tensor = tensor.contiguous();
        }
        
        // Synchronize device to ensure memory is ready
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaDeviceSynchronize failed: ") + cudaGetErrorString(err));
        }

        void* dev_ptr = tensor.data_ptr();
        std::cout << "dev_ptr: " << dev_ptr << std::endl;
        cudaIpcMemHandle_t handle;
        
        err = cudaIpcGetMemHandle(&handle, dev_ptr);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("cudaIpcGetMemHandle failed: ") + cudaGetErrorString(err));
        }

        // Convert handle to bytes
        return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t));
    }

    // New function to merge multiple tensors into one large tensor
    py::bytes merge_tensors(const std::vector<torch::Tensor>& tensors, int device) {
        // Step 1: Calculate the total number of elements across all tensors
        size_t total_size = 0;
        for (const auto& tensor : tensors) {
            total_size += tensor.numel();
        }

        // Step 2: Create an empty tensor to hold the merged data
        torch::Tensor merged_tensor = torch::empty({total_size}, torch::kFloat32).to(torch::kCUDA);

        // Step 3: Flatten each tensor and copy data into the merged tensor
        size_t current_offset = 0;  // Keep track of where to start copying data in the merged tensor
        for (const auto& tensor : tensors) {
            // Flatten the tensor and get a pointer to the flattened data
            auto flattened_tensor = tensor.flatten();

            // Copy the flattened data into the merged tensor
            merged_tensor.slice(0, current_offset, current_offset + flattened_tensor.numel()).copy_(flattened_tensor);

            // Update the current offset
            current_offset += flattened_tensor.numel();
        }

        return export_ipc_handle_pybind(merged_tensor,false);
    }

    py::bytes merge_tensors_and_export_ipc_handle(const std::vector<torch::Tensor>& tensors, int device) {
        // Step 1: Calculate the total number of elements across all tensors
        size_t total_size = 0;
        for (const auto& tensor : tensors) {
            total_size += tensor.numel();
        }

        // Step 2: Manually allocate memory using CUDA (without PyTorch allocation)
        void* dev_ptr = nullptr;
        cudaError_t err = cudaMalloc(&dev_ptr, total_size * sizeof(float));  // Adjust dtype size if necessary
        if (err != cudaSuccess) {
            throw std::runtime_error("CUDA memory allocation failed: " + std::string(cudaGetErrorString(err)));
        }

        // Step 3: Flatten each tensor and copy data into the allocated memory
        size_t current_offset = 0;  // Keep track of where to start copying data in the allocated memory
        for (const auto& tensor : tensors) {
            // Flatten the tensor and get a pointer to the flattened data
            auto flattened_tensor = tensor.flatten();
            
            // Copy the flattened data into the allocated memory
            err = cudaMemcpy(static_cast<char*>(dev_ptr) + current_offset, flattened_tensor.data_ptr(),
                            flattened_tensor.numel() * sizeof(float), cudaMemcpyDeviceToDevice);
            if (err != cudaSuccess) {
                cudaFree(dev_ptr);  // Clean up memory if something goes wrong
                throw std::runtime_error("CUDA memcpy failed: " + std::string(cudaGetErrorString(err)));
            }

            // Update the current offset
            current_offset += flattened_tensor.numel() * sizeof(float);
        }

        // Step 4: Create a tensor from the manually allocated memory
        // torch::Tensor merged_tensor = torch::from_blob(dev_ptr, {total_size}, torch::kCUDA, torch::TensorOptions().dtype(torch::kFloat32));

        // Step 5: Export the IPC handle for the merged tensor
        cudaIpcMemHandle_t handle;
        err = cudaIpcGetMemHandle(&handle, dev_ptr);
        if (err != cudaSuccess) {
            cudaFree(dev_ptr);  // Clean up memory if something goes wrong
            throw std::runtime_error("cudaIpcGetMemHandle failed: " + std::string(cudaGetErrorString(err)));
        }

        // Convert the handle to bytes and return
        return py::bytes(reinterpret_cast<const char*>(&handle), sizeof(cudaIpcMemHandle_t));
    }

    PYBIND11_MODULE(ipc_tensor_pybind, m) {
        m.doc() = "PyBind11 module for CUDA IPC tensor handling";
        
        m.def("open_ipc_handle", &open_ipc_handle_pybind, 
            "Open an IPC handle and return a device pointer",
            py::arg("handle_bytes"), py::arg("device"));
        
        m.def("create_tensor_from_ptr", &create_tensor_from_ptr,
            "Create a tensor from a device pointer",
            py::arg("dev_ptr"), py::arg("offset_bytes"),
            py::arg("shape"), py::arg("dtype"),
            py::arg("device"), py::arg("make_contiguous") = false);
        
        m.def("merge_tensors_and_export_ipc_handle", &merge_tensors_and_export_ipc_handle,
             "Merge multiple tensors and export the IPC handle",
            py::arg("tensors"), py::arg("device"));

        m.def("export_ipc_handle", &export_ipc_handle_pybind,
            "Export an IPC handle from a PyTorch tensor",
            py::arg("tensor"), py::arg("ensure_contiguous") = true);
    }