import numpy as np


class Net:

    def __init__(self, layers, num_epoch, batch_size=64, shuffle=False):
        self.layers = layers
        self.num_epoch = num_epoch
        self.batch_size = batch_size
        self.n_layers = len(layers)
        self.shuffle = shuffle

    def train(self, x_train, y_train, x_val, y_val, info=False, detailed_info=False):
        for epoch in range(self.num_epoch):
            if self.shuffle:
                x_y = np.c_[x_train.reshape(len(x_train), -1), y_train.reshape(len(y_train), -1)]
                np.random.shuffle(x_y)
                x_train = x_y[:, :x_train.size // len(x_train)].reshape(x_train.shape)
                y_train = x_y[:, x_train.size // len(x_train):].reshape(y_train.shape)
                
            batch_size = self.batch_size
            
            for i in range(int(x_train.shape[0] / batch_size)):
                
                x = x_train[i * batch_size:(i + 1) * batch_size]
                y = y_train[i * batch_size:(i + 1) * batch_size]
                
                if len(x) > 0:
                    x_layers = [x]
                    for k, layer in enumerate(self.layers):
                        if k < len(self.layers) - 1:
                            x_layers.append(layer.forward(x_layers[k]))
                        else:
                            x_layers.append(layer.forward(x_layers[k], y))
                    
                    g = [0] * (len(self.layers) + 1)
                    g[len(self.layers)] = y
                    
                    for k, layer in reversed(list(enumerate(self.layers))):  # not efficient, seems that creates a copy
                        g[k] = layer.backprop(x_layers[k], g[k + 1])
                    
                    if detailed_info and i == int(x_train.shape[0] / batch_size) - 1:
                        self._printInfo(g)
                        
            y_val_pred = self.predict(x_val)
            accuracy = len(y_val_pred[(y_val_pred - y_val) == 0]) / len(y_val_pred)
            
            if info:
                print(f'Epoch {epoch + 1} validation accuracy: {accuracy:.3f}')
            
            # implement dynamic reduction of learning rate
            if epoch % 10 == 10:
                for k, layer in enumerate(self.layers):
                    if hasattr(layer, 'l_r'):
                        layer.l_r /= 10.0

    def predict(self, x_test):
        x = [x_test]
        for k, layer in enumerate(self.layers):
            if k < len(self.layers) - 1:
                if hasattr(layer, 'train'):
                    layer.train = False
                x.append(layer.forward(x[k]))
            else:
                x.append(np.argmax(x[k], axis=1))
        return x.pop()

    def _printInfo(self, g):
        for k, layer in enumerate(self.layers):
            if hasattr(layer, 'W'):
                print(f'  Layer {k} Wmax, Wmin, bmax, bmin: {np.max(layer.W)}, {np.min(layer.W)}, {np.max(layer.b)}, {np.min(layer.b)}')
                print(f'     Gradient gmax, gmin, gmod: {np.max(g[k])}, {np.min(g[k])}, {np.average(np.sqrt(np.sum(g[k] * g[k], axis=1)))}')


class LinearLayer:

    def __init__(self, d_in, d_out, learn_rate=0.1, regu=0.0, drop=0.0):
        self.d_in = d_in
        self.d_out = d_out
        self.W = np.random.randn(d_in, d_out) * np.sqrt(2 / (d_in + d_out))
        self.b = np.random.randn(1, d_out) * np.sqrt(2 / (d_in + d_out))
        self.l_r = learn_rate
        self.regu = regu
        self.train = True
        self.drop = drop
        self.drop_mask = None

    def forward(self, x):
        for_out = x @ self.W + self.b * np.ones((x.shape[0], self.W.shape[1]))
        self.drop_mask = np.ones(for_out.shape)
        if self.train:
            self.drop_mask[:, (np.random.rand(for_out.shape[1])) < self.drop] = 0
            self.drop_mask *= 1.0 / (1.0 - self.drop)
        for_out = for_out * self.drop_mask
        return for_out

    def backprop(self, x, g):
        g = g * self.drop_mask
        back_out = g @ self.W.T
        self.W -= self.l_r * ((x.T @ g) + self.regu * self.W) / x.shape[0]
        self.b -= self.l_r * ((np.sum(g, axis=0)).reshape(1, -1)) / x.shape[0]  # no regularization in b
        return back_out


class ReLu:

    def __init__(self):
        self.active = None

    def forward(self, x):
        self.active = (x > 0).astype(int)
        for_out = x
        for_out[x < 0] = 0.0
        return for_out

    def backprop(self, x, g):
        back_out = g * self.active
        return back_out


class SoftMax:

    def __init__(self):
        self.z = None

    def forward(self, x, y):  ##y only in case we want to calculate loss
        X_max = x - np.amax(x, axis=1, keepdims=True)
        self.z = np.exp(X_max) / np.sum(np.exp(X_max), axis=1, keepdims=True)
        for_out = np.argmax(self.z, axis=1).reshape(x.shape[0], -1)
        return self.z

    def backprop(self, x, y):
        y_z = np.zeros(self.z.shape).flatten()
        y_z[y.astype(int).flatten() + np.arange(len(y)) * x.shape[1]] = 1.0
        y_z = y_z.reshape(self.z.shape)
        back_out = (self.z - y_z)
        return back_out
