from tqdm import tqdm
from torch import optim
import torch
import torch.nn as nn
import numpy as np
# from Toolkit import *
import torch.nn.functional as F
from matplotlib import pyplot as plt

seed = 0
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class predictNet(nn.Module):
    def __init__(self, kernel_size1, kernel_size2, data_length):
        super().__init__()
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        avglength = 151
        self.avgpool1 = nn.AvgPool1d(kernel_size=avglength, stride=1, padding=avglength//2)
        self.avgpool2 = nn.AvgPool1d(kernel_size=avglength, stride=1, padding=avglength//2)
        self.pred_length1 = 3
        self.pred_length2 = 21
        self.linear1 = nn.Sequential(
            nn.Linear(kernel_size1+1, 4*(kernel_size1+1), bias=True),
            nn.ReLU(0.2),
            nn.Linear(4*(kernel_size1+1), self.pred_length1, bias=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(kernel_size2+1, 4*(kernel_size2+1), bias=True),
            nn.ReLU(0.2),
            nn.Linear(4*(kernel_size2+1), self.pred_length2, bias=True)
        )
        self.avgpool3 = nn.AvgPool1d(kernel_size=self.pred_length1, stride=self.pred_length1)
        self.avgpool4 = nn.AvgPool1d(kernel_size=self.pred_length2, stride=self.pred_length2)
        
    def forward(self, output):
        # input shape : N 1 10000
        # # 卷积
        layernorm = torch.nn.LayerNorm(output.shape[-1]).to(device)
        output = layernorm(output.squeeze(1))
        output1 = output
        output2 = output

        avg1 = self.avgpool1(output1)
        avg2 = self.avgpool2(output2)
        energy = self.avgpool2(output1**2)
        # 移位拼接
        output1 = self.DIYroll(output1,self.kernel_size1)
        output2 = self.DIYroll(output2,self.kernel_size2)
        # 添加辅助趋势项
        output1 = torch.cat([output1, avg1],dim=0)
        output2 = torch.cat([output2, avg2],dim=0)
        # 线性变换 预测学习
        output1 = self.linear1(output1.t()).t()
        output2 = self.linear2(output2.t()).t()
        output1 = torch.flatten(output1.t()).unsqueeze(0)
        output2 = torch.flatten(output2.t()).unsqueeze(0)
        output1 = self.avgpool3(output1)
        output2 = self.avgpool4(output2)# 1 1 5600
        output1 = output1 / energy
        layernorm = torch.nn.LayerNorm(output1.shape[-1]).to(device)
        output1 = layernorm(output1)
        output2 = layernorm(output2)
        return output1.unsqueeze(1), output2.unsqueeze(1)
        
    def DIYroll(self, input, shift):
        output = input
        for i in range(1,shift):
            output = torch.cat([output, 
                                torch.roll(input,shifts=i,dims=1),
                                ],dim=0)
        return output
    



def trainPredictNet(*,model,data,T,lr=0.001,epoch=10):
    optimizer = optim.Adam(model.parameters(),lr=lr,weight_decay=1.0)
    criterion = nn.MSELoss()
    relu = torch.nn.ReLU()
    LossRecord = []
    data = data.data
    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        output1, output2 = model(data)
        loss = criterion(data+T, output1+output2+T) * 1 + torch.sum(output2[0][0] * output2[0][0]) * 100 + torch.sum(output2[0][0]) * torch.sum(output2[0][0]) * 100
        
        LossRecord.append(loss.item())
        loss.backward()
        optimizer.step()
    LossRecord = torch.tensor(LossRecord, device="cpu")
    plt.plot(LossRecord)
    plt.show()
    return model
    
