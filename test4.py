from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mlp import MLP, mse_loss, bce_loss

net = MLP(
    linear_1_in_features=100,
    linear_1_out_features=40,
    f_function='relu',
    linear_2_in_features=40,
    linear_2_out_features=200,
    g_function='sigmoid'
)
x = torch.randn(70, 100)
y = torch.randn(70, 200)

net.clear_grad_and_cache()
y_hat = net.forward(x)
J, dJdy_hat = mse_loss(y, y_hat)
net.backward(dJdy_hat)

#------------------------------------------------
# compare the result with autograd
net_autograd = nn.Sequential(
    OrderedDict([
        ('linear1', nn.Linear(100, 40)),
        ('relu1', nn.ReLU()),
        ('linear2', nn.Linear(40, 200)),
        ('sigmoid2', nn.Sigmoid()),
    ])
)
net_autograd.linear1.weight.data = net.parameters['W1']
net_autograd.linear1.bias.data = net.parameters['b1']
net_autograd.linear2.weight.data = net.parameters['W2']
net_autograd.linear2.bias.data = net.parameters['b2']

y_hat_autograd = net_autograd(x)

J_autograd = F.mse_loss(y_hat_autograd, y)

net_autograd.zero_grad()
J_autograd.backward()

print(J/J_autograd)
print(y_hat/y_hat_autograd)
print(net.grads['dJdW1'])
print(net_autograd.linear1.weight.grad.data)
print(net.grads['dJdb1'])
print(net_autograd.linear1.bias.grad.data)
print(net.grads['dJdW2'])
print(net_autograd.linear2.weight.grad.data)
print(net.grads['dJdb2'])
print(net_autograd.linear2.bias.grad.data)

print((J_autograd - J) < 1e-3)
print((y_hat_autograd - y_hat).norm() < 1e-3)
print((net_autograd.linear1.weight.grad.data - net.grads['dJdW1']).norm() < 1e-3)
print((net_autograd.linear1.bias.grad.data - net.grads['dJdb1']).norm() < 1e-3)
print((net_autograd.linear2.weight.grad.data - net.grads['dJdW2']).norm() < 1e-3)
print((net_autograd.linear2.bias.grad.data - net.grads['dJdb2']).norm()< 1e-3)
#------------------------------------------------
