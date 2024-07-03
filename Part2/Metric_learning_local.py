import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from tqdm import tqdm
import pandas as pd
from Toolkit import *
import torch.utils.data as Data
from United_model import *
from Toolkit import *

# 50维 只取20维做度量学习
# 将整个维度分为两份 对分别两份进行度量学习

torch.manual_seed(10)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Metric_Model(nn.Module):
    def __init__(self, input_data_dim):
        super(Metric_Model, self).__init__()
        self.input_data_dim = input_data_dim
        self.linear = nn.Linear(input_data_dim,input_data_dim)
        self.metric = nn.Sequential(
            nn.Linear(self.input_data_dim, 10000, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Linear(10000, 600, bias=False), # 原来50 0614修改
        )
        # 拉高度量学习维度，使得对比学习的输出结果可以将密度高的地方稀疏化（流形）

    def forward(self, output):
        output = self.linear(output)
        output = self.metric(output)
        return output

# 引入流形
# data 1000(50人*20) * 300
def train_Metric_Model(*, model, data, label, target, lr=0.0001, epoch=2):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1.0)
    dataset = Data.TensorDataset(data, label)
    loader = Data.DataLoader(dataset=dataset, batch_size=3, shuffle=True)
    criterion = nn.MSELoss()
    LossRecord = []
    length = data.shape[0]
    target_dis = (1 - target) * 1000 + 10*torch.ones(target.shape).to(device)  # 强制要求同类别间也有距离
    data = data.data

    for _ in tqdm(range(epoch)):
        optimizer.zero_grad()
        zero = torch.tensor(0.0).to(device)
        output = model(data)
        output = output.squeeze(1)
        # 度量学习 batch_size = 3
        save_dis = torch.zeros(data.shape[0],data.shape[0])
        loss2 = zero
        # 添加方差权重
        var_weight = torch.zeros(data.shape[0])
        persons = int(label[-1])
        one_persons = int(data.shape[0]//label[-1])
        pic_var = torch.zeros(persons)
        for i in range(persons):
            var_weight[one_persons*i:one_persons*(i+1)] = torch.sum(
                torch.var(output[one_persons*i:one_persons*(i+1),:],dim=0),
                dim=0)*100
            pic_var[i] = torch.sum(
                torch.var(output[one_persons*i:one_persons*(i+1),:],dim=0),
                dim=0)*100
        # 原始
        for i in range(data.shape[0]):
            sample = output[i].repeat(data.shape[0],1)
            save_dis = (sample - output)*(sample - output) # L2范数
            save_dis = torch.sum(save_dis,dim=1)
            save_dis = torch.where(save_dis>1050, 1050, save_dis)  # modify 0613 原本是注释掉
            loss2 = criterion(save_dis, target_dis[i]) + loss2 + var_weight[i] # 加方差
        loss = loss2
        loss.backward()
        optimizer.step()
        LossRecord.append(loss.item())
    return model

def Simloss(ans, target):
    U = ans * target  # 行和为正样本
    V = ans  # 行和为所有样本
    FU = torch.exp(U)
    FV = torch.exp(V)
    Usum = torch.sum(FU, dim=1)
    Vsum = torch.sum(FV, dim=1)
    output = -torch.log(Usum / Vsum)
    return torch.sum(output, dim=0)

############################

def run_Metric_Model(epoch, Pathlist):
    oneperson_begin = 0
    oneperson_end = 24
    oneperson_nums = oneperson_end - oneperson_begin
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin, oneperson_end=oneperson_end)[:,:,:].to(device)
    Unite_model = torch.load('Save_Model/United_model_device.pth').to(device).eval()
    feature1, ans, feature2 = Unite_model(data)
    features = feature2 # 对比学习
    data = features
    persons = int(data.shape[0]/oneperson_nums)

    # 标签
    label = torch.zeros(oneperson_nums*persons)
    for i in range(persons):
        label[i*oneperson_nums:(i+1)*oneperson_nums] = i
    # 目标余弦相似度矩阵
    target = torch.zeros(oneperson_nums*persons,oneperson_nums*persons)
    for i in range(persons):
        target[i*oneperson_nums:(i+1)*oneperson_nums, i*oneperson_nums:(i+1)*oneperson_nums] = 1

    data = data.to(device)
    target = target.to(device)

    # Metric_learning
    print('--------------度量学习-------------------')
    print('features shape :',features.shape)
    model = Metric_Model(input_data_dim=features.shape[-1]).to(device)
    model = train_Metric_Model(model=model,data=features,label=label,target=target,lr=0.0001,epoch=epoch)
    torch.save(model,'Save_Model/train_Metric_Model_local.pth')
    print('模型保存成功！')

    length = data.shape[0]
    ans = torch.zeros(length,length)

    output1 = model(data)
    output1 = output1.squeeze(1)
    print('度量学习结果形状：',output1.shape)
    ans = torch.zeros(data.shape[0],data.shape[0])
    for i in range(data.shape[0]):
        sample1 = output1[i].repeat(data.shape[0], 1)
        save_dis1 = (sample1 - output1) * (sample1 - output1)
        save_dis1 = torch.sum(save_dis1, dim=1)
        ans[i] = save_dis1

    ans = torch.where(ans > 1100, 1100, ans)  # modify 0613 原本是注释掉
    ans = -ans

    map = plt.imshow(ans.detach().numpy(), interpolation='nearest', cmap=cm.Blues, aspect='auto',)
    plt.colorbar(mappable=map,cax=None,ax=None,shrink=0.5)
