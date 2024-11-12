import mlx.core as mx

class Linear:
    def __init__(self, input_dims, output_dims, layer_id):
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.layer_id = layer_id

        k = mx.sqrt(1.0 / input_dims)
        self.weights = mx.random.uniform(-k, k, (output_dims, input_dims))
        self.bias = mx.zeros(output_dims)

    def forward(self, x):
        return mx.matmul(x, self.weights.T) + self.bias

    def update_layer(self, weights_grads, bias_grads, lr):
        self.weights -= weights_grads * lr
        self.bias -= bias_grads * lr


class MultiLayer:
    def __init__(self, input_size, hidden_sizes):
        self.input_size = input_size
        self.layers = []
        self.params = []

        for i, hidden_size in enumerate(hidden_sizes):
            if i == 0:
                self.layers.append(Linear(input_size, hidden_size, i))
            else:
                self.layers.append(Linear(hidden_sizes[i - 1], hidden_size, i))

        for layer in self.layers:
            self.params.append(layer.weights)
            self.params.append(layer.bias)
    def forward(self, x):
            mx.eval(x)
            y = x
            for i in range(len(self.layers)):
                if i == 0:
                    y = relu(self.layers[i].forward(y))
                else:
                    y = self.layers[i].forward(y)
                # self.print_vect(y)
            mx.eval(y)
            print("Y predictions:", y.item())
            return y
    def update(self, new_params):
        if len(new_params) != len(self.params):
            raise ValueError("Mismatch in parameter count")
        for i, layer in enumerate(self.layers):
            layer.weights = new_params[2 * i]
            layer.bias = new_params[2 * i + 1]
        self.params = new_params

    def update_parameters(self, grads, lr):
        if len(grads) != len(self.params):
            raise ValueError("Mismatch in gradient count")
        for i in range(len(self.params)):
            self.params[i] -= grads[i] * lr
        self.update(self.params)

ml = MultiLayer(2, [3, 3,1])

def relu(x):
    return mx.maximum(x, mx.array([0]))



x = mx.linspace(0,mx.pi,1000)
xp = mx.cos(x)
y = mx.sin(x) + 3*xp

x_tot = mx.transpose(mx.stack([x, xp]))
mx.eval(x_tot)

mx.eval(y)

def forward(params, input):
    x1 = mx.matmul(input, params[0].T) + params[1]
    x2 = relu(x1)

    x3 = mx.matmul(x2, params[2].T) + params[3]
    x4 = relu(x3)

    x5 = mx.matmul(x4, params[4].T) + params[5]
    return x5



def mse(params, input, y_true):
    outputs = forward(params, input)
    lvalue = mx.mean(mx.square(outputs - y_true))
    return lvalue

def mseP(params, inputs, targets):
    outputs = forward(params, inputs)
    lvalue = (outputs - targets).square().mean()
    return lvalue


for i in range(10000):
    params = ml.params
    random_index = mx.random.randint(0, 1000)

    # params.append([random_index])

    # value, grads = mx.value_and_grad(mse)(params, x_tot[random_index], y[random_index])

    # Returns lvalue, dlvalue/dparams
    lvalue, grads = mx.value_and_grad(mseP)(params, x_tot[random_index], y[random_index])
    print("Loss: ",lvalue.item())
    # print("Grads mean: ", mx.mean(grads[0]).item())

    ml.update_parameters(grads, 0.01)
