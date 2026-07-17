#include "engine.cu"
#include <vector>
#include <random>
#include <memory>
#include <algorithm>

class Module {
public:
    virtual ~Module() = default;
    virtual void zero_grad() {
        for (auto& p : parameters()) {
            if (p) p->zero_grad();
        }
    }
    virtual std::vector<std::shared_ptr<Tensor>> parameters() {
        return {};
    }
};

class Linear : public Module {
public:
    std::shared_ptr<Tensor> W;
    std::shared_ptr<Tensor> b;
    
    Linear(int nin, int nout) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<float> dist(0.0, 1.0);
        
        std::vector<float> w_data(nout * nin);
        for (auto& val : w_data) val = dist(gen);
        
        W = std::make_shared<Tensor>(std::vector<int>{nout, nin}, true);
        W->set_data(w_data);
        
        std::vector<float> b_data(nout, 0.0f);
        b = std::make_shared<Tensor>(std::vector<int>{nout}, true);
        b->set_data(b_data);
    }
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        Engine engine;
        auto wx = engine.mul(W, x);
        return engine.add(wx, b);
    }
    
    std::vector<std::shared_ptr<Tensor>> parameters() override {
        return {W, b};
    }
};

class MLP : public Module {
public:
    std::vector<std::shared_ptr<Linear>> layers;
    Engine engine;
    
    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());
        
        for (size_t i = 0; i < nouts.size(); ++i) {
            layers.push_back(std::make_shared<Linear>(sz[i], sz[i+1]));
        }
    }
    
    std::shared_ptr<Tensor> forward(const std::shared_ptr<Tensor>& x) {
        auto out = x;
        for (size_t i = 0; i < layers.size(); ++i) {
            out = layers[i]->forward(out);
            if (i != layers.size() - 1) {
                out = engine.relu(out);
            }
        }
        return out;
    }
    
    std::vector<std::shared_ptr<Tensor>> parameters() override {
        std::vector<std::shared_ptr<Tensor>> params;
        for (auto& layer : layers) {
            auto layer_params = layer->parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }
};
