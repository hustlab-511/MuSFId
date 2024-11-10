from TRM_Encoder_Decoder import *
from Toolkit import *
# from TRM_TRM_Unet import *
import torch
import torch.nn as nn
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pylab as plt

torch.manual_seed(10)

class United_Model(nn.Module):
    def __init__(self,seq_length,elementlength): # 6 50
        super().__init__()
        self.elementlength = elementlength
        self.seq_length = seq_length
        self.data_length = seq_length * 100 # 需要与TRM输出特征维度一致
        self.model1 = Transformer(seq_length=seq_length, elementlength=elementlength).to(device)
        self.model2 = Transformer_Encoder(
            input_data_dim=self.data_length,
            batches=5,
            each_batch_dim=int(self.data_length/5),
            feed_forward_hidden_dim=20,
        ).to(device)

    def forward(self, input):
        input = input.view(input.shape[0], int(input.shape[2] / self.elementlength), self.elementlength).detach()
        features, ans = self.model1(input)
        features = features.view(features.shape[0], 1, features.shape[1] * features.shape[2])
        ans = ans.view(ans.shape[0], 1, ans.shape[1] * ans.shape[2])
        output = self.model2(features)
        return features, ans, output.squeeze(1)

def Simloss(ans,target):
    U = ans * target # 行和为正样本
    V = ans # 行和为所有样本
    FU = torch.exp(U)
    FV = torch.exp(V)
    Usum = torch.sum(FU,dim=1)
    Vsum = torch.sum(FV,dim=1)
    output = -torch.log(Usum/Vsum)
    return torch.sum(output,dim=0)
def train_United_Model(*, model, data, origin, target, persons, elementlength, lr, epoch):
    optimizer1 = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    LossRecord = []
    for k in tqdm(range(epoch)):
        optimizer1.zero_grad()
        masked = data.clone()

        index = random.randint(0, data.shape[2] - elementlength)
        masked[:, :, index:index + elementlength] = 0
        index = random.randint(0, data.shape[2] - elementlength)
        masked[:, :, index:index + elementlength] = 0
        features, ans, output = model(masked)
        loss1 = criterion(ans, origin.detach())
        features = features.view(data.shape[0], features.shape[1] * features.shape[2])
        features = F.normalize(features)
        loss2 = criterion(torch.mm(features, features.t()), target) * 5000
        output = F.normalize(output)
        sim_ans = torch.mm(output, output.t())
        loss3 = criterion(sim_ans, target)
        
        loss = loss1 + loss2 + loss3# + Modloss

        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()
    return model

# 新增数据 1 9 16 21 25
def run_United_model(epoch, Pathlist):
    oneperson_begin = 0
    oneperson_end = 24
    elementlength = 50
    oneperson_nums = oneperson_end - oneperson_begin
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin, oneperson_end=oneperson_end) # N*oneperson_nums 1 900
    persons = int(data.shape[0] / oneperson_nums)
    # 某条数据为某个人的标签
    label = torch.zeros(oneperson_nums * persons)
    for i in range(persons):
        label[i * oneperson_nums:(i + 1) * oneperson_nums] = i
    
    # 目标余弦相似度矩阵
    target = torch.zeros(oneperson_nums * persons, oneperson_nums * persons)
    for i in range(persons):
        target[i * oneperson_nums:(i + 1) * oneperson_nums, i * oneperson_nums:(i + 1) * oneperson_nums] = 1

    origin = data.clone()

    model = United_Model(seq_length=int(data.shape[2]/elementlength),elementlength=elementlength) # 6 50


    # 随机掩码 掩码分成多个片段

    data = data.to(device)
    origin = origin.to(device)
    target = target.to(device)
    print('--------------掩码+对比学习-------------------')
    for i in range(1):
        model = train_United_Model(model=model, data=data, origin=origin, target=target, persons=persons, elementlength=elementlength, lr=0.001, epoch=epoch)

    torch.save(model,'Save_Model/United_model_device.pth')

    feature1, ans, feature2 = model(data)
    print(feature1.shape)
    print(ans.shape)
    print(feature2.shape)
    data = F.normalize(feature2)
    ans = torch.mm(data, data.t()).cpu()
    map = plt.imshow(ans.detach().numpy(), interpolation='nearest', cmap=cm.Blues, aspect='auto', )
    plt.colorbar(mappable=map, cax=None, ax=None, shrink=0.5)
