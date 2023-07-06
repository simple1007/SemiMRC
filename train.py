from transformers import BertModel, DistilBertModel, AdamW
# from torchsummary import summary as summary_
from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import random_split
from utils import MRCDataset, fileread
import torch.nn as nn
import torch
import torch.nn.functional as F
import transformers
transformers.logging.set_verbosity_error()
import pickle
with open('noun_model/w2i.pkl','rb') as f:
    w2i = pickle.load(f)
maxlen = 512#420
bert_model = DistilBertModel.from_pretrained('monologg/distilkobert', return_dict=False)
ht_flag = False
prefix = 'ht_token'
epoch = 10
batch_size = 50
reader = fileread('ht.csv')
s_epoch = 0
normal_tk = False
mrc_data = MRCDataset(reader,w2i,ht_flag,normal_tk)
train_size = 80000#int(80000 * 0.8)
validation_size = 10000#int(80000 * 0.1)#10000#len(mrc_data) - train_size
test = len(mrc_data) - validation_size - train_size

class BERTFineTune(nn.Module):
    def __init__(self,bert_model):
        super(BERTFineTune, self).__init__()
        self.bert_model = bert_model#DistilBertModel.from_pretrained('monologg/distilkobert', return_dict=False)
        self.srnn = nn.LSTM(input_size=768, hidden_size=100, num_layers = 1, batch_first=True, bidirectional = True)
        # self.ernn = nn.LSTM(input_size=768, hidden_size=100, num_layers = 1, batch_first=True, bidirectional = True)

        self.snn = nn.Linear(200,50)
        # self.enn = nn.Linear(200,50)

        self.dropout = nn.Dropout(0.3)
        self.snn2 = nn.Linear(50,1)
        self.enn2 = nn.Linear(50,1)
        # self.lstm = nn.LSTM(200)
        self.softmax1 = nn.LogSoftmax(dim=-1)
        # self.softmax2 = nn.LogSoftmax(dim=-1)
       
    def forward(self,input_ids,token_ids,attention_mask):
        # x = {'input_ids':input_ids,'token_type_ids':token_ids,'attention_mask':attention_mask}
        x = {'input_ids':input_ids,'attention_mask':attention_mask}
        x = self.bert_model(**x)
        sy,_ = self.srnn(x[0])
        # print(sy.shape)
        # exit()
        sy = self.snn(sy)
        # ey = self.enn(ey)

        sy = self.dropout(sy)
        # print(sy.shape)
        # ey = self.dropout(ey)
        ey = sy
        sy = self.snn2(sy)
        ey = self.enn2(ey)
        # print(sy.shape)
        sy = sy.squeeze(-1)
        ey = ey.squeeze(-1)
        # print(sy.shape)
        # exit()
        sy = self.softmax1(sy)
        ey = self.softmax1(ey)
        return sy,ey

class BERTHTFineTune(nn.Module):
    def __init__(self,bert_model,word_num):
        super(BERTHTFineTune, self).__init__()
        self.bert_model = bert_model
        self.embedding = nn.Embedding(num_embeddings=word_num, 
                            embedding_dim=64,
                            padding_idx=0)
        # self.lstm = nn.LSTM(64,128,batch_first=True)
        # self.linear = nn.Linear(896,900)
        
        self.srnn = nn.LSTM(input_size=768, hidden_size=100, num_layers = 1, batch_first=True, bidirectional = True)
        self.srnn2 = nn.LSTM(input_size=64, hidden_size=100, num_layers = 1, batch_first=True, bidirectional = True)
        self.snn = nn.Linear(99*2,128)
        # self.enn = nn.Linear(200,50)

        self.dropout = nn.Dropout(0.3)
        self.avgpool2d = nn.AvgPool1d(3,stride=2)
        self.avgpool2d2 = nn.AvgPool1d(3,stride=2)
        # self.flatten = nn.Flatten()
        # self.flatten_e = nn.Flatten()
        # self.avg = nn.AvgPool1d(30 0,stride=1)
        self.sout = nn.Linear(128,1)
        self.eout = nn.Linear(128,1)
        self.softmax1 = nn.LogSoftmax(dim=-1)
        # self.softmax2 = nn.LogSoftmax(dim=1)
        # input_tensor = {'input_ids':torch.tensor([[0]*512]),'token_type_ids':torch.tensor([[0]*512]),'attention_mask':torch.tensor([[0]*512])}
        # self.bert_layer = bert_model(**input_tensor)
        # print(type(self.bert_layer))
    def forward(self,input_ids,token_ids,attention_mask,ht):

        x = {'input_ids':input_ids,'attention_mask':attention_mask}
        x = self.bert_model(**x)[0]
        # print(x.shape)``
        # print(x)
        # exit()
        emb = self.embedding(ht)
        # lstm_res,_ = self.lstm(emb)
        # print(x.shape,lstm_res.shape)
        # exit()
        x,_ = self.srnn(x)
        
        xht,_ = self.srnn2(emb)

        x = self.avgpool2d(x)
        xht = self.avgpool2d2(xht)
        # print(x.shape,xht.shape)
        # exit()
        x = torch.cat([x,xht],dim=-1)
        # print(x.shape)
        # exit()
        # print(x.shape)
        # exit()
        x = self.snn(x)
        x = self.dropout(x)

        sy = self.sout(x)
        ey = self.eout(x)

        sy = sy.squeeze(-1)
        ey = ey.squeeze(-1)

        sy = self.softmax1(sy)
        ey = self.softmax1(ey)

        return sy,ey

def train(optimizer,scheduler,loss1,loss2,model,x,y,batch_size,device):
    model.train()
    # model.train()
    model.zero_grad()
    x[0] = x[0].to(device)
    x[1] = x[1].to(device)
    x[2] = x[2].to(device)
    
    # ht = ht.to(device)

    y[0] = y[0].to(device)
    y[1] = y[1].to(device)

    sy, ey = model(x[0],x[1],x[2])
    
    l1 = loss1(sy,y[0])
    l2 = loss2(ey,y[1])

    loss = (l1 + l2) / 2

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)

    optimizer.step()
    scheduler.step()

    return loss.item()# / batch_size

def trainht(optimizer,scheduler,loss1,loss2,model,x,ht,y,batch_size,device):
    model.train()
    # model.train()
    model.zero_grad()
    x[0] = x[0].to(device)
    x[1] = x[1].to(device)
    x[2] = x[2].to(device)
    
    ht = ht.to(device)
    # print(ht.shape)
    y[0] = y[0].to(device)
    y[1] = y[1].to(device)

    sy, ey = model(x[0],x[1],x[2],ht)
    
    l1 = loss1(sy,y[0])
    l2 = loss2(ey,y[1])

    loss = (l1 + l2) / 2

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)

    optimizer.step()
    scheduler.step()

    return loss.item()# / batch_size

def train2(optimizer,scheduler,loss1,loss2,model,x,y,batch_size,device):
    model.train()
    # model.train()
    optimizer.zero_grad()
    x[0] = x[0].to(device)
    x[1] = x[1].to(device)
    x[2] = x[2].to(device)
    
    y[0] = y[0].to(device)
    y[1] = y[1].to(device)
    y[2] = y[2].to(device)
    sy = model(x[0],x[1],x[2])
    sy = sy.view(sy.shape[0]*sy.shape[1],-1)
    y[2] = y[2].view(y[2].shape[0]*y[2].shape[1])
    # print(y[2].shape)
    # print(sy.shape)
    l1 = loss1(sy,y[2])
    # l2 = loss1(ey,y[1])

    loss = l1

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),max_grad_norm)

    optimizer.step()
    scheduler.step()

    return loss.item()# / batch_size


if __name__ == '__main__':
    from tqdm import tqdm
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # PyTorch v0.4.0
    # bert_model.to(device)
    # bert_model.eval()



    if ht_flag:
        model = BERTHTFineTune(bert_model,mrc_data.len_w2i())
    else:
        model = BERTFineTune(bert_model)
    # model.load_state_dict(torch.load('model/model'))
    model = model.to(device)

    # print(model)

    loss_s = nn.NLLLoss()
    loss_e = nn.NLLLoss()

    
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=5e-5)
    warmup_step = int( train_size * 0.1 )

    max_grad_norm = 1
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,num_training_steps=train_size * epoch)
    

    # test_size = len(mrc_data) - train_size - validation_size
    train_dataset, validation_dataset, _ = random_split(mrc_data, [train_size, validation_size,test])

    
    dataloader = torch.utils.data.DataLoader(dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    val_dataloader = torch.utils.data.DataLoader(dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True) 
        
    loss = 0
    result = 0
    result2 = 0
    count = 0
    vl_ = None
    vl_count = 0
    
    from tqdm import tqdm
    for e in range(epoch):
        running_loss = 0
        # model.train()
        
        # for index in tqdm(range(12910//batch_size)):
        for index, batch in enumerate(tqdm(dataloader)):
            x, ht, y = batch
            if ht_flag:
                loss__ = trainht(optimizer,scheduler,loss_s,loss_e,model,x,ht,y,batch_size,device)
            else:
                loss__ = train(optimizer,scheduler,loss_s,loss_e,model,x,y,batch_size,device)
            running_loss += loss__
            if index % 100 == 0 and index != 0:
                print(e,index,loss__,(running_loss / ((index+1))))

        model.eval()
        with torch.no_grad():
            count = 0
            
            scount = 0
            ecount = 0
            hhh = 0
            hhhh = 0
            total = 0
            vs = 0
            ve = 0
            for val in val_dataloader:
                x_,ht,y_ = val
                count+=1
                x_[0] = x_[0].to(device)
                x_[1] = x_[1].to(device)
                x_[2] = x_[2].to(device)
                ht = ht.to(device)
                y_[0] = y_[0].to(device)
                y_[1] = y_[1].to(device)
                y_[2] = y_[2].to(device)

                if ht_flag:
                    yy = model(x_[0],x_[1],x_[2],ht)
                else:
                    yy = model(x_[0],x_[1],x_[2])

                yy0 = yy[0]#.cpu().detach()
                yy1 = yy[1]#.cpu().detach()

                vl = (loss_s(yy[0],y_[0]) + loss_e(yy[1],y_[1])) / 2
                vs += vl.item()
                _,si = torch.topk(yy0,k=1,dim=-1)
                _,ei = torch.topk(yy1,k=1,dim=-1)
                
                eqs = torch.eq(y_[0],si)
                eqe = torch.eq(y_[1],ei)
                # torch.numel(a[a>1])
                preqs = eqs[eqs==True]
                preqe = eqe[eqe==True]

                scount += preqs.shape[0]
                ecount += preqe.shape[0]
      
            print(scount,ecount)
            print('vl',vs/(validation_size//batch_size))
            print('s,',scount/validation_size)
            print('e,',ecount/validation_size)
            cvl = vs/(validation_size//batch_size)
            # if vl_ == None:
            #     vl_ = cvl
            # elif vl_ < cvl:
            #     vl_count += 1
            #     if vl_count == 2:
            #         break
            # else:
            #     vl_ = cvl
            #     vl_count = 0
        torch.save(model.state_dict(), '{}/{}_{}'.format(prefix,prefix,e))#{}'.format(e))
    torch.save(model.state_dict(), '{}/{}'.format(prefix,prefix))
