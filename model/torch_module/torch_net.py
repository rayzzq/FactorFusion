from collections import namedtuple

import torch
import torch.nn as nn


class MlpNet(nn.Module):
    def __init__(self, input_dim, output_dim, layers=(256,), act="LeakyReLU"):
        super(MlpNet, self).__init__()    
        
        layers = [input_dim] + list(layers)
        dnn_layers = []
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        hidden_units = input_dim
        
        for i, (_input_dim, hidden_units) in enumerate(zip(layers[:-1], layers[1:])):
            fc = nn.Linear(_input_dim, hidden_units)
            if act == "LeakyReLU":
                activation = nn.LeakyReLU(negative_slope=0.1, inplace=False)
            elif act == "SiLU":
                activation = nn.SiLU()
            else:
                raise NotImplementedError(f"This type of input is not supported")
            bn = nn.BatchNorm1d(hidden_units)
            seq = nn.Sequential(fc, bn, activation)
            dnn_layers.append(seq)
        
        drop_input = nn.Dropout(0.05)
        dnn_layers.append(drop_input)
        
        fc = nn.Linear(hidden_units, output_dim)
        dnn_layers.append(fc)
        
        self.dnn_layers = nn.ModuleList(dnn_layers)
        self._weight_init()
        
        
    def _weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.1, mode="fan_in", nonlinearity="leaky_relu")
    
    def forward(self, x):
        cur_output = x
        for i, now_layer in enumerate(self.dnn_layers):
                cur_output = now_layer(cur_output)
        return cur_output
    
