import pandas as pd
import numpy as np
import ipdb
import glob
import os


excel = pd.read_csv('./demographics.csv')

patient_file = []
normal_file = []

for i in range(len(excel)): #excel 每一行数据
    file_list = glob.glob(os.path.join('./data', excel.iloc[i]['ID']+'*.txt')) #获取匹配的文件名列表

    if excel.iloc[i]['HoehnYahr'] == 0.0:   # 正常人
        print("normal:", excel.iloc[i]['ID'])
        normal_file += file_list
    else:                                   # 患者
        print("patient:", excel.iloc[i]['ID'])
        patient_file += file_list



for i,path in enumerate(patient_file):
    personal_data = np.loadtxt(path)
    personal_data = personal_data[:,[-2,-1]]
    np.savetxt('./data/small_data/patient{}.txt'.format(i),personal_data)


for i,path in enumerate(normal_file):
    personal_data = np.loadtxt(path)
    personal_data = personal_data[:,[-2,-1]]
    np.savetxt('./data/small_data/normal{}.txt'.format(i),personal_data)

ipdb.set_trace()
print('finished')


