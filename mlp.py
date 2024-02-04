import torch
import numpy as np

class MLP:
    def __init__(
        self,
        linear_1_in_features,
        linear_1_out_features,
        f_function,
        linear_2_in_features,
        linear_2_out_features,
        g_function
    ):
        """
        Args:
            linear_1_in_features: the in features of first linear layer
            linear_1_out_features: the out features of first linear layer
            linear_2_in_features: the in features of second linear layer
            linear_2_out_features: the out features of second linear layer
            f_function: string for the f function: relu | sigmoid | identity
            g_function: string for the g function: relu | sigmoid | identity
        """
        self.f_function = f_function
        self.g_function = g_function
        self.linear_1_in_features = linear_1_in_features
        self.linear_1_out_features = linear_1_out_features
        self.linear_2_in_features = linear_2_in_features
        self.linear_2_out_features = linear_2_out_features

        self.parameters = dict(
            W1 = torch.randn(linear_1_out_features, linear_1_in_features),
            b1 = torch.randn(linear_1_out_features),
            W2 = torch.randn(linear_2_out_features, linear_2_in_features),
            b2 = torch.randn(linear_2_out_features),
        )
        self.grads = dict(
            dJdW1 = torch.zeros(linear_1_out_features, linear_1_in_features),
            dJdb1 = torch.zeros(linear_1_out_features),
            dJdW2 = torch.zeros(linear_2_out_features, linear_2_in_features),
            dJdb2 = torch.zeros(linear_2_out_features),
        )

        # put all the cache value you need in self.cache
        self.cache = dict()

    def get_function(self, function_string):
        def identity(x):
            return x
        if function_string == 'relu':
            return torch.relu
        elif function_string == 'sigmoid':
            return torch.sigmoid
        else:
            return identity
    
    def get_derivative(self, function_string):
        def one(x):
            return torch.ones(x.size())
        def relu_derivative(x):
            return (x>=0).float()
        def sigmoid_derivative(x):
            return torch.exp(-x).mul(torch.sigmoid(x).mul(torch.sigmoid(x)))
        if function_string == 'relu':
            return relu_derivative
        elif function_string == 'sigmoid':
            return sigmoid_derivative
        else:
            return one

    def forward(self, x):
        """
        Args:
            x: tensor shape (batch_size, linear_1_in_features)
        """
        batch_size = x.size()[0]
        y_hat = torch.zeros(batch_size, self.linear_2_out_features)
        self.cache['x'] = x
        self.cache['s_1'] = torch.zeros(batch_size, self.linear_1_out_features)
        self.cache['a_1'] = torch.zeros(batch_size, self.linear_1_out_features)
        self.cache['s_2'] = torch.zeros(batch_size, self.linear_2_out_features)
        self.cache['y_hat'] = torch.zeros(batch_size, self.linear_2_out_features)
        f = self.get_function(self.f_function)
        g = self.get_function(self.g_function)
        for i in range(batch_size):
            x_batch = x[i]
            s_1 = self.parameters['W1'].matmul(x_batch) + self.parameters['b1']
            a_1 = f(s_1)
            s_2 = self.parameters['W2'].matmul(a_1) + self.parameters['b2']
            y_hat[i] = g(s_2)
            self.cache['s_1'][i] = s_1
            self.cache['a_1'][i] = a_1
            self.cache['s_2'][i] = s_2
            self.cache['y_hat'][i] = y_hat[i]
        # TODO: Implement the forward function
        return y_hat
    
    def backward(self, dJdy_hat):
        """
        Args:
            dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
        """
        # TODO: Implement the backward function
        batch_size = dJdy_hat.size()[0]
        k = dJdy_hat.size()[1]
        x = self.cache['x']
        s_1 = self.cache['s_1']
        a_1 = self.cache['a_1']
        s_2 = self.cache['s_2']
        y_hat = self.cache['y_hat']
        derivative_f = self.get_derivative(self.f_function)
        derivative_g = self.get_derivative(self.g_function)
        for i in range(batch_size):
            dy_ds_2 = torch.diag_embed(derivative_g(s_2[i]))
            da_1ds_1 = torch.diag_embed(derivative_f(s_1[i]))
            self.grads['dJdW2'] += (a_1[i].outer(dJdy_hat[i].matmul(dy_ds_2))).transpose(0,1)
            self.grads['dJdb2'] += dJdy_hat[i].matmul(dy_ds_2)
            self.grads['dJdW1'] += (x[i].outer(dJdy_hat[i].matmul(dy_ds_2.matmul(self.parameters['W2'].matmul(da_1ds_1))))).transpose(0,1)
            self.grads['dJdb1'] += dJdy_hat[i].matmul(dy_ds_2.matmul(self.parameters['W2'].matmul(da_1ds_1)))

    
    def clear_grad_and_cache(self):
        for grad in self.grads:
            self.grads[grad].zero_()
        self.cache = dict()

def mse_loss(y, y_hat):
    """
    Args:
        y: the label tensor (batch_size, linear_2_out_features)
        y_hat: the prediction tensor (batch_size, linear_2_out_features)

    Return:
        J: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the mse loss
    batch_size = y.size()[0]
    k = y.size()[1]     # (linear_2_out_features)
    loss = 0
    dJdy_hat = torch.zeros(batch_size, k)
    for i in range(batch_size):
            for j in range(k):
                loss += (y_hat[i][j] - y[i][j])**2
                dJdy_hat[i][j] = 2*(y_hat[i][j] - y[i][j])/(batch_size*k)
    loss /= batch_size*k
    return loss, dJdy_hat

def bce_loss(y, y_hat):
    """
    Args:
        y_hat: the prediction tensor
        y: the label tensor
        
    Return:
        loss: scalar of loss
        dJdy_hat: The gradient tensor of shape (batch_size, linear_2_out_features)
    """
    # TODO: Implement the bce loss
    batch_size = y.size()[0]
    k = y.size()[1]     # (linear_2_out_features)
    loss = 0
    dJdy_hat = torch.zeros(batch_size, k)
    for i in range(batch_size):
            for j in range(k):
                loss -= y[i][j]*np.log(y_hat[i][j])+(1-y[i][j])*np.log(1-y_hat[i][j])
                dJdy_hat[i][j] = ((y_hat[i][j]-y[i][j])/(y_hat[i][j]*(1-y_hat[i][j])))/(batch_size*k)
    loss /= batch_size*k
    return loss, dJdy_hat











