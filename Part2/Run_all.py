from United_model import *
from FingerPrint_5_quick import *
from testModel import *

Pathlist = [
    'X1001',
    'X1002',
    'X1003',
    'X1004',
    'X1005',
    'X1006',
    'X1007',
    'X1008', # p
    'X1009',
    'X10010', # p
]

for i in range(Pathlist.__len__()):
    Pathlist[i] = 'DataSet/new_'+Pathlist[i]+'_onebeats.pth'

batches = 3
run_United_model(500, Pathlist)
run_Metric_Model(500, Pathlist)
run_FingerPrint(500, Pathlist, batches=batches)

run_quick_test_ans(Pathlist, batches=batches)