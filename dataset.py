import glob
import os
import numpy as np
import ipdb
from torch.utils.data import Dataset,TensorDataset
import torch



data_length = 512
data = []
label = []

## 数据需要归一化
## 要不要考虑不要开始一段时间内数据

normal_file_list = glob.glob('./data/small_data/normal*.txt')
patient_file_list = glob.glob('./data/small_data/patient*.txt')

for path in normal_file_list:
    personal_data = np.loadtxt(path)
    startid = 0
    endid = data_length
    while endid <= len(personal_data):
        temp_data = personal_data[startid:endid,:]
        temp_label = 0
        data.append(temp_data)
        label.append(temp_label)
        startid = endid
        endid += data_length

for path in normal_file_list:
    personal_data = np.loadtxt(path)
    startid = 0 + int(data_length/2)
    endid = data_length + int(data_length/2)
    while endid <= len(personal_data):
        temp_data = personal_data[startid:endid,:]
        temp_label = 0
        data.append(temp_data)
        label.append(temp_label)
        startid = endid
        endid += data_length



for path in patient_file_list:
    personal_data = np.loadtxt(path)
    startid = 0
    endid = data_length
    while endid <= len(personal_data):
        temp_data = personal_data[startid:endid,:]
        temp_label = 1
        data.append(temp_data)
        label.append(temp_label)
        startid = endid
        endid += data_length


index = np.random.permutation(len(data))
data = np.array(data)
label = np.array(label)

data = data[index,:,:]
label = label[index]

data_length = len(data)
valid_slice= slice(0,int(data_length*0.15))
test_slice= slice(int(data_length*0.15),int(data_length*0.15)*2)
train_slice = slice(int(data_length*0.15)*2, data_length)

valid_data = data[valid_slice]
test_data = data[test_slice]
train_data = data[train_slice]

valid_label = label[valid_slice]
test_label = label[test_slice]
train_label = label[train_slice]


valid_data = torch.tensor(valid_data)
test_data = torch.tensor(test_data)
train_data = torch.tensor(train_data)

valid_label = torch.tensor(valid_label)
test_label = torch.tensor(test_label)
train_label = torch.tensor(train_label)

trainset = TensorDataset(train_data,train_label)
validset = TensorDataset(valid_data,valid_label)
testset = TensorDataset(test_data,test_label)

torch.save(trainset,'./trainset.pt')
torch.save(validset,'./validset.pt')
torch.save(testset,'./testset.pt')

ipdb.set_trace()
print('finished')

