import torch
import matplotlib.pylab as plt
import pandas as pd
import numpy as np
import torch.nn.functional as F
from matplotlib import cm
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
import random
from scipy import signal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 生成ResUnet所需数据
def get_ResUnet_data(*,Pathlist,oneperson_begin,oneperson_end):
    # data = torch.zeros(1, 1, 900)
    data = torch.load(Pathlist[0])
    data = torch.zeros(1, 1, data.shape[-1])
    for it in Pathlist:
        data1 = torch.load(it)
        data1 = data1[oneperson_begin:oneperson_end,:,:]
        data = torch.cat([data,data1],dim=0)
    return data[1:,:,:]

def splitDataSet(data, label, persons, oneperson_nums): # 要求输入 shape: N x
    mid = int(oneperson_nums / 3 * 2)
    print("mid =",mid,"oneperson_nums =",oneperson_nums)
    X_train = data[0:mid,:]
    X_test = data[mid:oneperson_nums,:]
    y_train = label[0:mid]
    y_test = label[mid:oneperson_nums]
    for i in range(1,persons):
        X_train = torch.cat([X_train, data[oneperson_nums*i:oneperson_nums*i+mid,:]],dim=0)
        X_test = torch.cat([X_test, data[oneperson_nums*i+mid:oneperson_nums*i+oneperson_nums,:]],dim=0)
        y_train = torch.cat([y_train, label[oneperson_nums*i:oneperson_nums*i+mid]],dim=0)
        y_test = torch.cat([y_test, label[oneperson_nums*i+mid:oneperson_nums*i+oneperson_nums]],dim=0)
    return X_train, X_test, y_train, y_test
    
# 输入数据生成模板
def create_Muban(Pathlist, test_index):
    oneperson_begin = 0
    oneperson_end = 36
    focus_nums = 24 - oneperson_begin
    oneperson_nums = oneperson_end - oneperson_begin
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin+test_index, oneperson_end=oneperson_end+test_index)[:,:,:].to(device)
    Unite_model = torch.load('Save_Model/United_model_device.pth', map_location=device).eval()
    feature1, ans, feature2 = Unite_model(data)
    features = feature2
    data = features
    persons = int(data.shape[0]/oneperson_nums)
    # 标签
    label = torch.zeros(oneperson_nums*persons)
    for i in range(persons):
        label[i*oneperson_nums:(i+1)*oneperson_nums] = i
    # 目标分类矩阵
    target = torch.zeros(oneperson_nums*persons,persons)
    for i in range(persons):
        target[i*oneperson_nums:(i+1)*oneperson_nums, i:(i+1)] = 100
    data = data.to(device)
    # Metric_learning
    Metric_model = torch.load('Save_Model/train_Metric_Model_local.pth', map_location=device).eval()

    output1 = Metric_model(data)
    output1 = output1.squeeze(1).cpu()

    muban = torch.zeros(persons, output1.shape[-1])
    # 通道一
    output = output1
    for i in range(persons):
        muban[i] = torch.mean(output[i * oneperson_nums:i * oneperson_nums + focus_nums, :], dim=0)
    return muban, label
