//
//  NN.hpp
//  tutorial
//
//  Created by Christophe Prat on 10/09/2024.
//

#ifndef NN_hpp
#define NN_hpp

#include <set>
#include <stdio.h>
#include "game.h"
// #include <mlx/mlx.h>
const int NUM_WEIGHTS = 6;
const int NUM_LAYERS = 2;

// using namespace mlx::core;
#include <iostream>
#include <vector>
#include <set>
#include <cmath>
#include <functional>

class Value {
public:
    double data;
    double grad;
    std::vector<Value*> _prev;
    std::string _op;
    std::function<void()> _backward;

    Value(double data, const std::vector<Value*>& _children = {}, const std::string& _op = "")
        : data(data), grad(0), _prev(_children), _op(_op), _backward([](){}) {}

    Value operator+(const Value& other) {
        Value out(this->data + other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "+");

        out._backward = [out, this, other]() mutable {
            this->grad += out.grad;
            const_cast<Value&>(other).grad += out.grad;
        };

        return out;
    }

    Value operator*(const Value& other) {
        Value out(this->data * other.data, {const_cast<Value*>(this), const_cast<Value*>(&other)}, "*");

        out._backward = [out, this, other]() mutable {
            this->grad += other.data * out.grad;
            const_cast<Value&>(other).grad += this->data * out.grad;
        };

        return out;
    }

    Value operator^(double other) {
        Value out(std::pow(this->data, other), {const_cast<Value*>(this)}, "**" + std::to_string(other));

        out._backward = [out, this, other]() mutable {
            this->grad += (other * std::pow(this->data, other - 1)) * out.grad;
        };

        return out;
    }

    Value relu() {
        Value out(this->data > 0 ? this->data : 0, {const_cast<Value*>(this)}, "ReLU");

        out._backward = [out, this]() mutable {
            this->grad += (out.data > 0) * out.grad;
        };

        return out;
    }

    void backward() {
        std::vector<Value*> topo;
        std::set<Value*> visited;

        std::function<void(Value*)> build_topo = [&](Value* v) {
            if (visited.find(v) == visited.end()) {
                visited.insert(v);
                for (Value* child : v->_prev) {
                    build_topo(child);
                }
                topo.push_back(v);
            }
        };

        build_topo(this);

        this->grad = 1;
        for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
            (*it)->_backward();
        }
    }

    Value operator-() {
        return *this * -1;
    }

    Value operator+(double other) {
        return *this + Value(other);
    }

    // Value operator-(const Value& other) {
    //     return *this + (-other);
    // }

    // Value operator/(const Value& other) {
    //     return *this * (other ^ -1);
    // }

    Value operator*(double other) {
        return *this * Value(other);
    }

    Value operator/(double other) {
        return *this * (Value(other) ^ -1);
    }

    friend std::ostream& operator<<(std::ostream& os, const Value& v) {
        os << "Value(data=" << v.data << ", grad=" << v.grad << ")";
        return os;
    }
};

class Module {
public:
    virtual void zero_grad() {
        for (Value* p : parameters()) {
            p->grad = 0;
        }
    }

    virtual std::vector<Value*> parameters() {
        return {};
    }
};

class Neuron : public Module {
public:
    std::vector<Value> w;
    Value b;
    bool nonlin;



    Neuron(int nin, bool nonlin = true) : nonlin(nonlin), b(0) {
        std::srand(std::time(0));
        for (int i = 0; i < nin; ++i) {
            w.push_back(Value(static_cast<double>(std::rand()) / RAND_MAX * 2 - 1));
        }
        b = Value(0);
    }

    Value operator()(const std::vector<Value>& x) {
        // Ensure the input size matches the number of weights
        // assert(x.size() == w.size() && "Input size must match the number of weights");

        Value act = b; // Start with the bias
        for (size_t i = 0; i < w.size(); ++i) {
            act = act + (w[i] * x[i]);
        }
        return nonlin ? act.relu() : act;
    }

    std::vector<Value*> parameters() override {
        std::vector<Value*> params;
        for (Value& v : w) {
            params.push_back(&v);
        }
        params.push_back(&b);
        return params;
    }

    std::string toString() const {
        return std::string(nonlin ? "ReLU" : "Linear") + "Neuron(" + std::to_string(w.size()) + ")";
    }
};

class Layer : public Module {
public:
    std::vector<Neuron> neurons;

    Layer(int nin, int nout, bool nonlin = true) {
        for (int i = 0; i < nout; ++i) {
            neurons.push_back(Neuron(nin, nonlin));
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& x) {
        std::vector<Value> out;
        for (Neuron& n : neurons) {
            out.push_back(n(x));
        }
        return out.size() == 1 ? std::vector<Value>{out[0]} : out;
    }

    std::vector<Value*> parameters() override {
        std::vector<Value*> params;
        for (Neuron& n : neurons) {
            std::vector<Value*> neuron_params = n.parameters();
            params.insert(params.end(), neuron_params.begin(), neuron_params.end());
        }
        return params;
    }

    std::string toString() const {
        std::string result = "Layer of [";
        for (size_t i = 0; i < neurons.size(); ++i) {
            result += neurons[i].toString();
            if (i < neurons.size() - 1) {
                result += ", ";
            }
        }
        result += "]";
        return result;
    }
};

class MLP : public Module {
public:
    std::vector<Layer> layers;

    MLP(int nin, const std::vector<int>& nouts) {
        std::vector<int> sz = {nin};
        sz.insert(sz.end(), nouts.begin(), nouts.end());
        for (size_t i = 0; i < nouts.size(); ++i) {
            layers.push_back(Layer(sz[i], sz[i + 1], i != nouts.size() - 1));
        }
    }

    std::vector<Value> operator()(const std::vector<Value>& x) {
        std::vector<Value> out = x;
        for (Layer& layer : layers) {
            out = layer(out);
        }
        return out;
    }

    std::vector<Value*> parameters() override {
        std::vector<Value*> params;
        for (Layer& layer : layers) {
            std::vector<Value*> layer_params = layer.parameters();
            params.insert(params.end(), layer_params.begin(), layer_params.end());
        }
        return params;
    }

    std::string toString() const {
        std::string result = "MLP of [";
        for (size_t i = 0; i < layers.size(); ++i) {
            result += layers[i].toString();
            if (i < layers.size() - 1) {
                result += ", ";
            }
        }
        result += "]";
        return result;
    }
};

bool tickCallback(mat* m, block* s, block* nextBl, evars* e, unsigned int* score, MLP ml, unsigned int index, bool userMode, block** BASIC_BLOCKS);
MLP initMLP();
#endif /* NN_hpp */
