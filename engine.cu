#include <cuda_runtime.h>
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <stdexcept>
#include <iostream>
#include <cmath>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__global__ void add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] + b[idx];
}

__global__ void mul_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] * b[idx];
}

__global__ void exp_kernel(const float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = expf(a[idx]);
}

__global__ void relu_kernel(const float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = a[idx] > 0 ? a[idx] : 0;
}

__global__ void relu_grad_kernel(const float* a, const float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_in[idx] = (a[idx] > 0) ? grad_out[idx] : 0;
}

__global__ void sigmoid_kernel(const float* a, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = 1.0f / (1.0f + expf(-a[idx]));
}

__global__ void sigmoid_grad_kernel(const float* out, const float* grad_out, float* grad_in, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) grad_in[idx] = grad_out[idx] * out[idx] * (1.0f - out[idx]);
}

__global__ void add_grad_kernel(const float* grad_a, const float* grad_b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) out[idx] = grad_a[idx] + grad_b[idx];
}

__global__ void fill_kernel(float* data, float val, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = val;
}

class Tensor {
public:
    std::vector<int> shape;
    int size;
    float* data;
    float* grad;
    bool requires_grad;
    std::shared_ptr<Tensor> grad_accum;
    
    Tensor(const std::vector<int>& shape_, bool requires_grad_ = false) 
        : shape(shape_), requires_grad(requires_grad_) {
        size = 1;
        for (int dim : shape) size *= dim;
        
        CUDA_CHECK(cudaMalloc(&data, size * sizeof(float)));
        if (requires_grad) {
            CUDA_CHECK(cudaMalloc(&grad, size * sizeof(float)));
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            fill_kernel<<<blocks, threads>>>(grad, 0.0f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
        } else {
            grad = nullptr;
        }
    }
    
    Tensor(const Tensor& other) = delete;
    Tensor& operator=(const Tensor& other) = delete;
    
    ~Tensor() {
        if (data) CUDA_CHECK(cudaFree(data));
        if (grad) CUDA_CHECK(cudaFree(grad));
    }
    
    void set_data(const std::vector<float>& host_data) {
        if (host_data.size() != size) {
            throw std::runtime_error("Data size mismatch");
        }
        CUDA_CHECK(cudaMemcpy(data, host_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    std::vector<float> get_data() const {
        std::vector<float> host_data(size);
        CUDA_CHECK(cudaMemcpy(host_data.data(), data, size * sizeof(float), cudaMemcpyDeviceToHost));
        return host_data;
    }
    
    void zero_grad() {
        if (requires_grad && grad) {
            int threads = 256;
            int blocks = (size + threads - 1) / threads;
            fill_kernel<<<blocks, threads>>>(grad, 0.0f, size);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
};

class Function {
public:
    virtual ~Function() = default;
    virtual std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) = 0;
    virtual std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) = 0;
    std::vector<std::shared_ptr<Tensor>> saved_tensors;
};

class Add : public Function {
public:
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override {
        if (inputs.size() != 2) throw std::runtime_error("Add requires 2 inputs");
        saved_tensors = inputs;
        
        auto out = std::make_shared<Tensor>(inputs[0]->shape, inputs[0]->requires_grad);
        int n = out->size;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        add_kernel<<<blocks, threads>>>(inputs[0]->data, inputs[1]->data, out->data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
    
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override {
        std::vector<std::shared_ptr<Tensor>> grads;
        for (auto& inp : saved_tensors) {
            if (inp->requires_grad) {
                auto grad_in = std::make_shared<Tensor>(inp->shape, true);
                int n = inp->size;
                int threads = 256;
                int blocks = (n + threads - 1) / threads;
                add_grad_kernel<<<blocks, threads>>>(grad_output->data, grad_in->data, grad_in->grad, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                grads.push_back(grad_in);
            } else {
                grads.push_back(nullptr);
            }
        }
        return grads;
    }
};

class Mul : public Function {
public:
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override {
        if (inputs.size() != 2) throw std::runtime_error("Mul requires 2 inputs");
        saved_tensors = inputs;
        
        auto out = std::make_shared<Tensor>(inputs[0]->shape, inputs[0]->requires_grad);
        int n = out->size;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        mul_kernel<<<blocks, threads>>>(inputs[0]->data, inputs[1]->data, out->data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
    
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override {
        std::vector<std::shared_ptr<Tensor>> grads;
        for (auto& inp : saved_tensors) {
            if (inp->requires_grad) {
                auto grad_in = std::make_shared<Tensor>(inp->shape, true);
                int n = inp->size;
                int threads = 256;
                int blocks = (n + threads - 1) / threads;
                mul_kernel<<<blocks, threads>>>(grad_output->data, 
                    (inp == saved_tensors[0]) ? saved_tensors[1]->data : saved_tensors[0]->data,
                    grad_in->grad, n);
                CUDA_CHECK(cudaDeviceSynchronize());
                grads.push_back(grad_in);
            } else {
                grads.push_back(nullptr);
            }
        }
        return grads;
    }
};

class Exp : public Function {
public:
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override {
        if (inputs.size() != 1) throw std::runtime_error("Exp requires 1 input");
        saved_tensors = inputs;
        
        auto out = std::make_shared<Tensor>(inputs[0]->shape, inputs[0]->requires_grad);
        int n = out->size;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        exp_kernel<<<blocks, threads>>>(inputs[0]->data, out->data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
    
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override {
        std::vector<std::shared_ptr<Tensor>> grads;
        if (saved_tensors[0]->requires_grad) {
            auto grad_in = std::make_shared<Tensor>(saved_tensors[0]->shape, true);
            int n = grad_in->size;
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            auto exp_out = std::make_shared<Tensor>(saved_tensors[0]->shape, false);
            exp_kernel<<<blocks, threads>>>(saved_tensors[0]->data, exp_out->data, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            mul_kernel<<<blocks, threads>>>(grad_output->data, exp_out->data, grad_in->grad, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            grads.push_back(grad_in);
        } else {
            grads.push_back(nullptr);
        }
        return grads;
    }
};

class ReLU : public Function {
public:
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override {
        if (inputs.size() != 1) throw std::runtime_error("ReLU requires 1 input");
        saved_tensors = inputs;
        
        auto out = std::make_shared<Tensor>(inputs[0]->shape, inputs[0]->requires_grad);
        int n = out->size;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        relu_kernel<<<blocks, threads>>>(inputs[0]->data, out->data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        return out;
    }
    
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override {
        std::vector<std::shared_ptr<Tensor>> grads;
        if (saved_tensors[0]->requires_grad) {
            auto grad_in = std::make_shared<Tensor>(saved_tensors[0]->shape, true);
            int n = grad_in->size;
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            relu_grad_kernel<<<blocks, threads>>>(saved_tensors[0]->data, grad_output->data, grad_in->grad, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            grads.push_back(grad_in);
        } else {
            grads.push_back(nullptr);
        }
        return grads;
    }
};

class Sigmoid : public Function {
public:
    std::shared_ptr<Tensor> forward(const std::vector<std::shared_ptr<Tensor>>& inputs) override {
        if (inputs.size() != 1) throw std::runtime_error("Sigmoid requires 1 input");
        saved_tensors = inputs;
        
        auto out = std::make_shared<Tensor>(inputs[0]->shape, inputs[0]->requires_grad);
        int n = out->size;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        sigmoid_kernel<<<blocks, threads>>>(inputs[0]->data, out->data, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        saved_tensors.push_back(out);
        return out;
    }
    
    std::vector<std::shared_ptr<Tensor>> backward(const std::shared_ptr<Tensor>& grad_output) override {
        std::vector<std::shared_ptr<Tensor>> grads;
        if (saved_tensors[0]->requires_grad) {
            auto grad_in = std::make_shared<Tensor>(saved_tensors[0]->shape, true);
            int n = grad_in->size;
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            sigmoid_grad_kernel<<<blocks, threads>>>(saved_tensors[1]->data, grad_output->data, grad_in->grad, n);
            CUDA_CHECK(cudaDeviceSynchronize());
            grads.push_back(grad_in);
        } else {
            grads.push_back(nullptr);
        }
        return grads;
    }
};

class Engine {
public:
    std::unordered_map<std::shared_ptr<Tensor>, std::shared_ptr<Function>> grad_fn;
    
    std::shared_ptr<Tensor> add(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
        auto fn = std::make_shared<Add>();
        auto out = fn->forward({a, b});
        if (out->requires_grad) {
            grad_fn[out] = fn;
        }
        return out;
    }
    
    std::shared_ptr<Tensor> mul(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
        auto fn = std::make_shared<Mul>();
        auto out = fn->forward({a, b});
        if (out->requires_grad) {
            grad_fn[out] = fn;
        }
        return out;
    }
    
    std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {
        auto fn = std::make_shared<Exp>();
        auto out = fn->forward({a});
        if (out->requires_grad) {
            grad_fn[out] = fn;
        }
        return out;
    }
    
    std::shared_ptr<Tensor> relu(const std::shared_ptr<Tensor>& a) {
        auto fn = std::make_shared<ReLU>();
        auto out = fn->forward({a});
        if (out->requires_grad) {
            grad_fn[out] = fn;
        }
        return out;
    }
    
    std::shared_ptr<Tensor> sigmoid(const std::shared_ptr<Tensor>& a) {
        auto fn = std::make_shared<Sigmoid>();
        auto out = fn->forward({a});
        if (out->requires_grad) {
            grad_fn[out] = fn;
        }
        return out;
    }
    
    void backward(const std::shared_ptr<Tensor>& loss) {
        if (!loss->requires_grad) {
            throw std::runtime_error("Loss does not require gradients");
        }
        
        loss->zero_grad();
        int n = loss->size;
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        fill_kernel<<<blocks, threads>>>(loss->grad, 1.0f, n);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        std::vector<std::shared_ptr<Tensor>> topo;
        std::unordered_map<std::shared_ptr<Tensor>, bool> visited;
        
        std::function<void(std::shared_ptr<Tensor>)> build_topo = [&](std::shared_ptr<Tensor> v) {
            if (visited[v]) return;
            visited[v] = true;
            if (grad_fn.find(v) != grad_fn.end()) {
                for (auto& inp : grad_fn[v]->saved_tensors) {
                    if (inp->requires_grad) {
                        build_topo(inp);
                    }
                }
            }
            topo.push_back(v);
        };
        
        build_topo(loss);
        
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            auto& v = *it;
            if (grad_fn.find(v) != grad_fn.end()) {
                auto grads = grad_fn[v]->backward(std::make_shared<Tensor>(*v));
                for (size_t i = 0; i < grads.size(); ++i) {
                    if (grads[i] && grads[i]->requires_grad) {
                        auto& inp = grad_fn[v]->saved_tensors[i];
                        if (inp->requires_grad) {
                            int n2 = inp->size;
                            int threads2 = 256;
                            int blocks2 = (n2 + threads2 - 1) / threads2;
                            add_grad_kernel<<<blocks2, threads2>>>(grads[i]->grad, inp->grad, inp->grad, n2);
                            CUDA_CHECK(cudaDeviceSynchronize());
                        }
                    }
                }
            }
        }
    }
};
