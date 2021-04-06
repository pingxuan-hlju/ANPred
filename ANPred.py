import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn.functional as F
import random
drug_num=763
dis_num=681
w1,w2=0.01,0.01
w_d1,w_d2=1,0.1
left_num_epoch=100
right_num_epoch=150
a=0.2
skip_t=5
skip_m=6
def readfile1():
    drug_dis_mat=np.loadtxt('../data/drug_dis_mat.txt')
    drugdis_train = np.loadtxt('../data/train_data.txt')
    drug_sim=np.loadtxt('../data/drug_similarity.txt')
    dis_sim=np.loadtxt('../data/disease_similarity.txt')
    train_id=np.loadtxt('../data/train_index.txt')
    np.random.shuffle(train_id)
    test_id1=np.loadtxt('../data/test_index.txt')
    return drug_dis_mat,drug_sim,dis_sim,train_id,test_id1,drugdis_train

def readfile2():
    drug_dis_mat = np.loadtxt('../data/drug_dis_mat.txt')
    drugdis_train=np.loadtxt ('../data/train_data.txt')
    drug_sim = np.loadtxt('../data/drug_similarity.txt')
    dis_sim = np.loadtxt('../data/disease_similarity.txt')
    test_id1 = np.loadtxt('../data/test_index1.txt')
    return drug_dis_mat,drug_sim,dis_sim,test_id1,drugdis_train


def randm_work(M,xsim,train_id):
    Sp = np.zeros((M, M))
    for i in range(763):
        sort_score = sorted(enumerate(xsim[i]), key=lambda x: x[1], reverse=True)
        for j in range(5):
            Sp[i, sort_score[j][0]] = 1
            Sp[sort_score[j][0], i] = 1
    node_candition = []
    for i in range(763):
        ll = np.nonzero(Sp[i])[0].tolist()
        node_candition.append(ll)
    node_walk = []
    for epoch in range(skip_t):
        for d in range(len(train_id[:, 0])):
            list_node = []
            m = int(train_id[d, 0])
            list_node.append(m)
            for i in range(skip_m-1):
                candition = int(random.choice(node_candition[m]))
                list_node.append(candition)
                m = candition
            list_node.append(train_id[d, 1])
            node_walk.append(list_node)
    node_walk1 = np.array(node_walk)
    return node_walk1

def get_target(words, idx, window_size=8):
    R=window_size
    start = idx - R if (idx - R) > 0 else 0
    stop = idx + R
    target_words = words[start:idx] + words[idx + 1:stop + 1]
    return list(target_words)


def get_batches(words, batch_size, window_size=8):
    n_batches = len(words) // batch_size
    words = words[:n_batches * batch_size]

    for idx in range(0, len(words), batch_size):
        x, y = [], []
        batch = words[idx:idx + batch_size]
        for ii in range(1):
            batch_x = batch[ii]
            batch_y = get_target(batch, ii, window_size)
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y

def node_target(M,xsim,train_id):
    node_node_walk=randm_work(M,xsim,train_id)
    node_list = []
    node_neg_list = []
    dis_list = []
    for i in range(node_node_walk.shape[0]):
        words = node_node_walk[i].tolist()
        words1 = words[:-2]
        xx = []
        for x, y in get_batches(words1, batch_size=len(words1)):
            node_list.extend(x)
            node_neg_list.extend(y)
            xx.extend([words[-1]] * len(x))
            dis_list.extend(xx)
    node_target_mari = np.array([node_list, node_neg_list, dis_list]).T
    return node_target_mari

def load_data(train_index,drug_sim,dis_sim,drugdis_train0,BATCH_SIZE):
    x = []
    y = []
    for j in range(train_index.shape[0]):
        temp_save=[]
        x_A = int(train_index[j][0])
        y_A = int(train_index[j][1])
        drug_sim_dis=np.concatenate((drug_sim[x_A], drugdis_train0[x_A]),axis=0)
        dis_sim_drug=np.concatenate((drugdis_train0.T[y_A], dis_sim[y_A]),axis=0)
        temp_save.append(drug_sim_dis)
        temp_save.append(dis_sim_drug)
        label = drugdis_train0[[x_A], [y_A]]
        x.append([temp_save])
        y.append(label)
    x = torch.FloatTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    torch_dataset = Data.TensorDataset(x, y)
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    return data2_loader

def load_data1(train_index,train_index1,drug_sim,dis_sim,drugdis_train0,BATCH_SIZE):
    x = []
    y = []
    v=[]
    w=[]
    cc=[]
    for j in range(train_index.shape[0]):
        temp_save1=[]
        temp_save = []
        x_A = int(train_index[j][0])
        y_A = int(train_index[j][1]) 
        v_A=int(train_index[j][2]) 
        cc_A=int(train_index1[j][1])
        drug_sim_dis=np.concatenate((drug_sim[x_A], drugdis_train0[x_A]),axis=0)  
        dis_sim_drug=np.concatenate((drugdis_train0.T[v_A], dis_sim[v_A]),axis=0)
        temp_save.append(drug_sim_dis)
        temp_save1.append(dis_sim_drug)

        label = drugdis_train0[[x_A], [v_A]]
        x.append([temp_save])   
        y.append(y_A)           
        v.append([temp_save1])  
        cc.append(cc_A)        
        w.append(label)         
    x = torch.FloatTensor(np.array(x))
    y = torch.LongTensor(np.array(y))
    v = torch.FloatTensor(np.array(v))
    cc = torch.LongTensor(np.array(cc))
    w = torch.LongTensor(np.array(w))
    torch_dataset = Data.TensorDataset(x, y,v,cc,w)
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    return data2_loader

def load_data2(train_index,drug_sim,dis_sim,drugdis_train0,BATCH_SIZE):
    x = []
    y = []
    z=[]
    for j in range(train_index.shape[0]):
        temp_save=[]
        temp_save1 = []
        x_A = int(train_index[j][0])
        y_A = int(train_index[j][1]) 
        drug_sim_dis=np.concatenate((drug_sim[x_A], drugdis_train0[x_A]),axis=0)  
        dis_sim_drug=np.concatenate((drugdis_train0.T[y_A], dis_sim[y_A]),axis=0)
        temp_save.append(drug_sim_dis)
        temp_save1.append(dis_sim_drug)
        label = drugdis_train0[[x_A], [y_A]]
        x.append([temp_save])
        y.append([temp_save1])
        z.append(label)
    x = torch.FloatTensor(np.array(x))
    y = torch.FloatTensor(np.array(y))
    z = torch.LongTensor(np.array(z))
    torch_dataset = Data.TensorDataset(x, y,z) 
    data2_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )
    return data2_loader

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3,20), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1,2), stride=1))#16,1,2,722
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(3,20), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1))
        self.fc = nn.Sequential(
            nn.Linear(45056, 2000),  # 22912
            nn.ReLU(),
            nn.Linear(2000, 2),)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = out.view(x.size(0), -1)  # reshape
        out = self.fc(out)
        return out

class Auto_drug(nn.Module):
    def __init__(self):
        super(Auto_drug,self).__init__()

        self.encoder=nn.Sequential(
            nn.Linear(1444,500),
            nn.ReLU(True),
            nn.Linear(500,100),
            #nn.ReLU(True),
        )
        self.decoder=nn.Sequential(
            nn.Linear(100,500),
            nn.ReLU(True),
            nn.Linear(500,1444),
            nn.Tanh()
        )
    def forward_once(self,x):
        x=self.encoder(x)
        y=self.decoder(x)
        return x,y
    def forward(self, x):
        drug_en,drug_de=self.forward_once(x)
        return drug_en,drug_de


class SkipGram_drug(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super(SkipGram_drug,self).__init__()
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        scores = self.output(x)
        log_ps = self.log_softmax(scores)
        return log_ps

class Auto_dis(nn.Module):
    def __init__(self):
        super(Auto_dis,self).__init__()

        self.encoder=nn.Sequential(
            nn.Linear(1444,500),
            nn.ReLU(True),
            nn.Linear(500,100),
            #nn.ReLU(True),
        )
        self.decoder=nn.Sequential(
            nn.Linear(100,500),
            nn.ReLU(True),
            nn.Linear(500,1444),
            nn.Tanh()
        )
    def forward_once(self,x):
        x=self.encoder(x)
        y=self.decoder(x)
        return x,y
    def forward(self, x):
        dis_en,dis_de=self.forward_once(x)

        return dis_en,dis_de

class SkipGram_dis(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super(SkipGram_dis,self).__init__()
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)
    def forward(self, x):
        scores = self.output(x)
        log_ps = self.log_softmax(scores)
        return log_ps

class class_fic(nn.Module):
    def __init__(self):
        super(class_fic, self).__init__()
        self.con_layer = nn.Sequential(
             nn.Conv2d(1,16,kernel_size=2,stride=1,padding=1),  
             nn.ReLU(),
             nn.MaxPool2d(kernel_size=(2,2),stride=2),                
         )
        self.fc = nn.Sequential(
            #nn.Linear(200, 50),
            nn.Linear(1600,100),
            nn.ReLU(),
            nn.Linear(100,2)
        )

    def forward(self, x):
        x = self.con_layer(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

class AUTO_Loss(nn.Module):
    def __init__(self):
        super(AUTO_Loss, self).__init__()
    def forward(self,x1,y1,lable):
        loss=torch.norm(y1 - x1, 2).pow(2)
        return loss

drug_dis_mat,drug_sim,dis_sim,train_id,test_id1,train_data=readfile1()
left_train_loader = load_data(train_id, drug_sim,dis_sim,train_data, 10)
left_test_loader_all=load_data(test_id1, drug_sim,dis_sim, drug_dis_mat,763)
cnn=CNN()
optimizer=torch.optim.Adam(cnn.parameters(),lr=0.0005)
loss_func=nn.CrossEntropyLoss()
for epoch in range(left_num_epoch):
    train_loss=0
    train_acc=0
    for step, (x, train_label) in enumerate(left_train_loader):
        b_x = Variable(x)
        train_label=Variable(train_label.squeeze(1))
        out=cnn(b_x)
        loss = loss_func(out, train_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item ()
        _, pred = out.max(1)
        num_correct = (pred == train_label).sum().item()
        num_correct = num_correct
        acc = num_correct / b_x.shape[0]
        train_acc += acc
        if step % 100 == 0:
            print('Epoch: ', epoch, '| train loss: %.8f' % loss.data.numpy())#.cpu().numpy())
    print('Epoch: {}, Train Loss: {:.8f}, Train Acc: {:.6f}'
          .format(epoch, train_loss / len(left_train_loader), train_acc / len(left_train_loader)))

cnn.eval()
left_out=np.zeros((0,2))

for test_x,test_label in left_test_loader_all:
    test_x=Variable(test_x)
    test_label = Variable(test_label.squeeze(1))
    test_out = cnn(test_x)
    test_out=F.softmax(test_out,dim=1)
    _,pred_y = test_out.max(1)
    left_out=np.vstack((left_out,test_out.detach().numpy()))
print('left_out',left_out)


drug_dis_mat,drug_sim,dis_sim,test_id1,drugdis_train=readfile2()
train_skip=node_target(drug_num,drug_sim,train_id)
train_skip_disease=node_target(dis_num,dis_sim,train_id)
right_train_loader = load_data1(train_skip, train_skip_disease,drug_sim,dis_sim,drugdis_train, 10)
right_test_loader=load_data2(test_id1, drug_sim,dis_sim, drug_dis_mat,763)

model_drug = Auto_drug()
model_skip=SkipGram_drug(drug_num,100)
criterion2 = nn.NLLLoss()
criterion1=AUTO_Loss()
optimizer1 = torch.optim.Adam(model_drug.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(model_skip.parameters(), lr=0.001)
model_dis = Auto_dis()
model_skip_dis=SkipGram_dis(dis_num,100)
criterion_dis2 = nn.NLLLoss()
criterion_dis1=AUTO_Loss()
optimizer_dis1 = torch.optim.Adam(model_dis.parameters(), lr=0.001)
optimizer_dis2 = torch.optim.Adam(model_skip_dis.parameters(), lr=0.001)

for e in range(right_num_epoch):
    steps=0
    auto_skip_loss=0
    auto_loss1=0
    auto_loss2=0
    mull=np.zeros((0, 100))
    mull1 = np.zeros((0, 100))
    label_neu=np.zeros((0, 1))
    # get input and target batches
    for input_r,input_r_target,input_d,_,label in right_train_loader:

        steps += 1
        input_r= Variable(input_r.view(input_r.size(0), -1))
        input_d  = Variable(input_d .view(input_d .size(0), -1))
        label = label.squeeze(1)

        r_en,r_de=model_drug(input_r)
        regula_loss=0
        for pam in model_drug.parameters() :
            regula_loss+=torch.sum(abs(pam).pow(2))
        loss1 = criterion1(r_de, input_r)+w1*regula_loss
        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        x_r,_,_,_=model_drug(input_r)
        log_ps=model_skip(x_r)
        loss2=w2*criterion2(log_ps,input_r_target)
        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        auto_skip_loss = auto_skip_loss+loss1.item()+loss2.item()
        auto_loss1+=loss1.item()
        auto_loss2+=loss2.item()

        r_en = (r_en.view(r_en .size(0), -1)).detach().numpy()
        label = (label.view(label.size(0), -1)).detach().numpy()

        mull = np.vstack((mull, r_en))
        label_neu = np.vstack((label_neu, label))

    x = torch.FloatTensor(mull)
    y = torch.LongTensor(label_neu)
    torch_dataset = Data.TensorDataset(x, y)
    neu_loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

for e in range(right_num_epoch):
    steps=0
    auto_skip_loss=0
    auto_loss1=0
    auto_loss2=0
    mull=np.zeros((0, 100))
    mull1 = np.zeros((0, 100))
    label_neu=np.zeros((0, 1))
    # get input and target batches
    for input_r,_,input_d,input_d_target,label in right_train_loader:

        steps += 1
        input_r= Variable(input_r.view(input_r.size(0), -1))
        input_d  = Variable(input_d .view(input_d .size(0), -1))
        label = label.squeeze(1)

        r_en,r_de,d_en,d_de=model_dis(input_r,input_d)
        regula_loss=0
        for pam in model_dis.parameters():
            regula_loss+=torch.sum(abs(pam).pow(2))
        loss1 = criterion_dis1(d_de, input_d)+w_d1*regula_loss
        optimizer_dis1.zero_grad()
        loss1.backward()
        optimizer_dis1.step()

        x_d,_=model_dis(input_d)
        log_ps=model_skip_dis(x_d)
        loss2=w_d2*criterion_dis2 (log_ps,input_d_target)
        optimizer_dis2.zero_grad()
        loss2.backward()
        optimizer_dis2.step()

        auto_skip_loss = auto_skip_loss+loss1.item()+loss2.item()
        auto_loss1+=loss1.item()
        auto_loss2+=loss2.item()
        r_en = (r_en.view(r_en .size(0), -1)).detach().numpy()
        d_en = (d_en.view(d_en .size(0), -1)).detach().numpy()
        label = (label.view(label.size(0), -1)).detach().numpy()

        mull1 = np.vstack((mull1, d_en))
        label_neu=np.vstack((label_neu,label))

    x = torch.FloatTensor(mull1)
    y = torch.LongTensor(label_neu)
    torch_dataset = Data.TensorDataset(x,y)
    neu_loader1 = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=0,
        drop_last=True
    )

neural=class_fic()
optimizer3 = torch.optim.Adam(neural.parameters(), lr=0.001)
neural_loss = nn.CrossEntropyLoss()

for epoch in range(right_num_epoch):
    neu_loss=0
    neu_cor=0
    for ((c_r,y),(c_d,_))in zip(neu_loader,neu_loader1):

        b_x=torch.cat((c_r,c_d),1).reshape(10,1,2,100)
        b_x = Variable(b_x)
        b_y=Variable(y.squeeze(1))
        output=neural(b_x)
        loss=neural_loss(output,b_y)
        optimizer3.zero_grad()
        loss.backward()
        optimizer3.step()
        neu_loss+=loss.item()

        _, pred = output.max(1)
        num_correct = (pred == b_y).sum().item()
        neu_cor+=num_correct
    print('Epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}'
              .format(epoch, neu_loss / len(neu_loader), neu_cor ))


model_drug.eval()
mull=np.zeros((0, 100))
#mul=np.zeros((0, 200))
label_neu=np.zeros((0, 1))
for test_x,test_y, label in right_test_loader:
    test_x = Variable(test_x)
    test_y = Variable(test_y)
    label = Variable(label.squeeze(1))
    r_en,r_de=model_drug(test_x)
    r_en = (r_en.view(r_en.size(0), -1)).detach().numpy()
    label =(label .view(label .size(0), -1)).detach().numpy()
    # (10,200)
    #encoder = np.hstack((r_en, d_en))  
    mull = np.vstack((mull, r_en))
    label_neu = np.vstack((label_neu, label))

x = torch.FloatTensor(mull)
y = torch.LongTensor(label_neu)
# y = torch.LongTensor(np.array(label_neu))
torch_dataset = Data.TensorDataset(x, y)
neu_test_loader = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=763,
    shuffle=False,
    num_workers=0,
    drop_last=True
)

model_dis.eval()
mull=np.zeros((0, 100))
mull1=np.zeros((0, 100))
#mul=np.zeros((0, 200))
label_neu=np.zeros((0, 1))
for test_x,test_y, label in right_test_loader:
    test_x = Variable(test_x)
    test_y = Variable(test_y)
    label = Variable(label.squeeze(1))
    d_en,d_co =model_dis(test_y)
    d_en = (d_en.view(d_en.size(0), -1)).detach().numpy()
    label =(label .view(label .size(0), -1)).detach().numpy()
    mull1 = np.vstack((mull1, d_en))
    label_neu = np.vstack((label_neu, label))

x = torch.FloatTensor(mull1)
y = torch.LongTensor(label_neu)
torch_dataset = Data.TensorDataset( x, y)
neu_test_loader1 = Data.DataLoader(
    dataset=torch_dataset,
    batch_size=763,
    shuffle=False,
    num_workers=0,
    drop_last=True
)

neural.eval()
test_acc=0
num_cor=0
right_out = np.zeros((0, 2))
for ((test_x_0,test_label0),(test_y1,_)) in zip(neu_test_loader,neu_test_loader1):
    test_x = torch.cat((test_x_0, test_y1), 1).reshape(763,1,2,100)
    test_x = Variable(test_x)
    test_label = test_label0.squeeze(1)
    test_label = Variable(test_label)

    test_out = neural(test_x)
    test_out=F.softmax(test_out,dim=1)
    _,pred_y = test_out.max(1)
    num_correct = (pred_y == test_label).sum().item()
    acc = num_correct / test_x.shape[0]
    test_acc += acc
    num_cor += num_correct
    right_out=np.vstack((right_out,test_out.detach().cpu().numpy()))
print('right_out',right_out)

output=a*left_out +(1-a)*right_out
