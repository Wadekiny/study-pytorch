import torch
from torch import nn
import net
from torch.utils.data import Dataset,TensorDataset
import ipdb


EPOCHS =  20
BS = 128
LR = 0.001


# dataset
trainset = torch.load("./trainset.pt")
validset = torch.load("./validset.pt")
testset = torch.load("./testset.pt")

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=BS, shuffle=True)
valid_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=BS, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=testset, batch_size=BS, shuffle=False)

valid_loader = torch.utils.data.DataLoader(dataset=validset+testset, batch_size=BS, shuffle=False)

# model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = net.ResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = LR)




def train():
    model.train()
    for epoch in range(EPOCHS):
        print('-- epochs: {}/{}'.format(epoch,EPOCHS))
        for i,(data,label) in enumerate(train_loader):
            data = data.reshape(-1,1,2,512).float().to(device)
            label = label.to(device)
            

            output = model(data)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            
            if i%10 == 0:
                print("EPoch [{}/{}], step [{}/{}] trainLoss: {:.4f}"
                      .format(epoch+1, EPOCHS, i+1, len(train_loader), loss.item()))
            # if i == len(train_loader) -1:
            #     print(output)
            #     print(label)

        torch.save(model,"./model/model{}.pt".format(epoch)) 
        valid(model)

def valid(model):
    model.eval()
    with torch.no_grad():
        right_count = [0,0] #cls0,1 预测正确
        label_count = [0,0] #cls0,1 标签个数
        loss_all = 0
        for i,(data,label) in enumerate(valid_loader):
            data = data.reshape(-1,1,2,512).float().to(device)
            label = label.to(device)
            
            output = model(data)
            loss = criterion(output, label)
            loss_all += loss.item()

            outclass = torch.argmax(output,dim=1)
            for i,la in enumerate(label):
                label_count[la] += 1
                if label[i] == outclass[i]: #预测正确
                    right_count[la] += 1
        loss_avg = loss_all/len(valid_loader)
        print((loss_avg, right_count, label_count))
        return loss_avg, right_count, label_count

# def test(model):
#     model.eval()
#     with torch.no_grad():
#         right_count = [0,0] #cls0,1 预测正确
#         label_count = [0,0] #cls0,1 标签个数
#         loss_all = 0
#         for i,(data,label) in enumerate(test_loader):
#             data = data.reshape(-1,1,2,512).float().to(device)
#             label = label.to(device)
#             
#             output = model(data)
#             loss = criterion(output, label)
#             loss_all += loss.item()
#
#             outclass = torch.argmax(output,dim=1)
#             for i,la in enumerate(label):
#                 label_count[la] += 1
#                 if label[i] == outclass[i]: #预测正确
#                     right_count[la] += 1
#         loss_avg = loss_all/len(test_loader)
#         print((loss_avg, right_count, label_count))
#         return loss_avg, right_count, label_count




train()
ipdb.set_trace()







