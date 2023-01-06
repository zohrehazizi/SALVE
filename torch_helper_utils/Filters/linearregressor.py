import torch.nn as nn 
import torch

class LConvRegressor(nn.Conv2d):
    def __init__(self, kernel_size=5, eps=None):
        assert kernel_size%2==1, "kernel size should be an odd number"
        padding = int((kernel_size-1)//2)
        super().__init__(in_channels=3, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='reflect', bias=True)
        self.eps = eps
    def set_weights(self, L_reg):
        k = self.kernel_size[0]
        weight = torch.tensor(L_reg.coef_.reshape(1,1,k,k))*torch.tensor([0.2989, 0.5870, 0.1140]).reshape(1,3,1,1)
        bias = torch.tensor(L_reg.intercept_)
        self.weight.data = weight.to(torch.float32)
        self.bias.data = bias.to(torch.float32)
    def forward(self, x):
        x = super().forward(x)
        if self.eps is not None:
            x = torch.clip(x, min=self.eps)
        return x

class RConvRegressor(nn.Conv2d):
    def __init__(self, kernel_size=5, eps=None):
        assert kernel_size%2==1, "kernel size should be an odd number"
        padding = int((kernel_size-1)//2)
        super().__init__(in_channels=1, out_channels=1, kernel_size=kernel_size, padding=padding, padding_mode='reflect', bias=True)
        self.eps = eps
    def set_weights(self, R_reg):
        k = self.kernel_size[0]
        weight = torch.tensor(R_reg.coef_.reshape(1,1,k,k))
        bias = torch.tensor(R_reg.intercept_)
        self.weight.data = weight.to(torch.float32)
        self.bias.data = bias.to(torch.float32)
    def forward(self, x):
        x = super().forward(x)
        if self.eps is not None:
            x = torch.clip(x, min=self.eps)
        return x