#include "engine.cu"
#include <vector>
#include <memory>

class Optimizer {
public:
    std::vector<std::shared_ptr<Tensor>> params;
    float lr;
    
    Optimizer(const std::vector<std::shared_ptr<Tensor>>& params_, float lr_ = 0.001f)
        : params(params_), lr(lr_) {
        validate_params();
    }
    
    void validate_params() {
        for (auto& p : params) {
            if (!p) {
                throw std::runtime_error("Invalid tensor pointer");
            }
        }
    }
    
    void step() {
        for (auto& p : params) {
            if (!p->grad) continue;
            
            int n = p->size;
            int threads = 256;
            int blocks = (n + threads - 1) / threads;
            
            update_kernel<<<blocks, threads>>>(p->data, p->grad, lr, n);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
    }
    
private:
    static __global__ void update_kernel(float* data, const float* grad, float lr, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] -= lr * grad[idx];
        }
    }
};
