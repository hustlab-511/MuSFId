from FingerPrint_5_quick import *
from United_model import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def run_quick_test_ans(Pathlist, batches):
    print('测试集结果：') 
    train_Pathlist = [
        'X1001',
        'X1002',
        'X1003',
        'X1004',
        'X1005',
        'X1006',
        'X1007',
        'X1008',
        'X1009',
        'X10010',
    ]

    for i in range(train_Pathlist.__len__()):
        train_Pathlist[i] = 'DataSet/new_'+train_Pathlist[i]+'_onebeats.pth'
    muban, label1 = create_Muban(train_Pathlist, 0)
    oneperson_begin = 0
    oneperson_end = 36
    focus_nums = 24 - oneperson_begin 
    oneperson_nums = oneperson_end - oneperson_begin
    
    data = get_ResUnet_data(Pathlist=Pathlist, oneperson_begin=oneperson_begin, oneperson_end=oneperson_end)[:,:,:].to(device)
    print("data.shape = ", data.shape)
    persons = int(data.shape[0]/oneperson_nums)
    # 标签
    label = torch.zeros(oneperson_nums*persons)
    for i in range(persons):
        label[i*oneperson_nums:(i+1)*oneperson_nums] = i
    
    X_train, X_test, y_train, y_test = splitDataSet(data, label, persons, oneperson_nums)
    # 对比学习
    Unite_model = torch.load('Save_Model/United_model_device.pth', map_location=device).eval()
    feature1, ans, feature2 = Unite_model(X_test)
    features = feature2
    data = features
    data = data.to(device)
    # Metric_learning
    Metric_model = torch.load('Save_Model/train_Metric_Model_local.pth', map_location=device).eval()
    output1 = Metric_model(data)  # 度量学习
    output1 = output1.squeeze(1)
    output1 = output1.to(device)
    # 通道一
    output = output1
    oneperson_nums = oneperson_end-focus_nums # X_test测试集每人的数据大小
    test_data = X_test
    ans = torch.zeros(test_data.shape[0], len(train_Pathlist)).to(device)
    muban = muban.to(device)
    for i in range(test_data.shape[0]):
        sample = output[i].repeat(len(train_Pathlist), 1).to(device)
        save_dis = (sample - muban) * (sample - muban) # L2范数
        save_dis = torch.sum(save_dis, dim=1)
        ans[i] = save_dis

    print('ans.shape=',ans.shape)
    
    # 构建序列增强数据
    # batches = 4
    aans = torch.zeros(1,batches*ans.shape[-1]).to(device)
    from sklearn.model_selection import train_test_split
    for i in range(ans.shape[0]):
        # 随机生成batches-1条数据
        if batches==1:
            aans = torch.cat([aans.to(device), ans.to(device)], dim=0)
            break
        else:
            databegin = i//oneperson_nums*oneperson_nums
            _, X_test, _, _ = train_test_split(ans[databegin:databegin+oneperson_nums,:],
                                               ans[databegin:databegin+oneperson_nums,:],
                                               test_size=(batches-1)/oneperson_nums,
                                               random_state=i)
            mid = torch.cat([ans[i:i+1,:].to(device), X_test.to(device)],dim=0)
            mid = mid.view(1,batches*ans.shape[-1])
            aans = torch.cat([aans.to(device), mid], dim=0)
    ans = aans[1:,:]
    ##################
    ans = -ans.to(device)
    model = torch.load('Save_Model/FingerPrint_quick_1.pth', map_location=device).eval()
    print('模型读取成功！')

    # Seq条连续序列测试结果
    right = 0
    low = 0
    refuse = 0
    false = 0
    record = torch.zeros(train_Pathlist.__len__(), 3)
    softmax= nn.Softmax(dim=1)
    relu = nn.ReLU()
    
    confusion_metric = torch.zeros(train_Pathlist.__len__(),train_Pathlist.__len__()) # 横轴为真实值  纵轴为预测值
    print(ans.shape)
    for i in range(ans.shape[0]):
        output = model(ans[i:i+1,:]) # 输出 1 60
        output = relu(output)/torch.sum(relu(output), dim=1)
        value, index = torch.max(output, dim=1)
        confusion_metric[int(y_test[i])][int(index.cpu())] += 1
    print(confusion_metric)
    acc = 0
    for i in range(train_Pathlist.__len__()):
        acc += confusion_metric[i][i]
    print("Acc = ",acc / (train_Pathlist.__len__()*(oneperson_end-focus_nums)))
    precision = 0
    for i in range(train_Pathlist.__len__()):
        precision += confusion_metric[i][i] / (torch.sum(confusion_metric[i:i+1,:]))
    print("precision = ",precision / train_Pathlist.__len__())
    recall = 0
    for i in range(train_Pathlist.__len__()):
        recall += confusion_metric[i][i] / (torch.sum(confusion_metric[:,i:i+1]))
    print("recall = ",recall / train_Pathlist.__len__())