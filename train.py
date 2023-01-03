from transformers import BertModel, DistilBertModel, AdamW
# from torchsummary import summary as summary_
from collections import defaultdict
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import random_split
from utils import MRCDataset, fileread
import torch.nn as nn
import torch
import torch.nn.functional as F

maxlen = 420
bert_model = DistilBertModel.from_pretrained('monologg/distilkobert', return_dict=False)
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

        sy = self.snn(sy)
        # ey = self.enn(ey)

        sy = self.dropout(sy)
        # ey = self.dropout(ey)
        ey = sy
        sy = self.snn2(sy)
        ey = self.enn2(ey)

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
    model = BERTFineTune(bert_model)
    # model.load_state_dict(torch.load('span_model'))
    model = model.to(device)

    # print(model)

    
    loss_s = nn.NLLLoss()
    loss_e = nn.NLLLoss()

    reader = fileread('data.csv')
    mrc_data = MRCDataset(reader)
    train_size = int(len(mrc_data) * 0.8)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=5e-5)
    warmup_step = int( train_size * 0.1 )
    epoch = 4
    batch_size = 20
    max_grad_norm = 1
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,num_training_steps=train_size * epoch)
    validation_size = len(mrc_data) - train_size
    test = len(mrc_data) - validation_size - train_size
    # test_size = len(mrc_data) - train_size - validation_size
    train_dataset, validation_dataset = random_split(mrc_data, [train_size, validation_size])

    
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
    
    for e in range(epoch):
        running_loss = 0
        # model.train()
        
        # for index in tqdm(range(12910//batch_size)):
        for index, batch in enumerate(dataloader):
            x, y = batch
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
                x_,y_ = val
                count+=1
                x_[0] = x_[0].to(device)
                x_[1] = x_[1].to(device)
                x_[2] = x_[2].to(device)
                
                y_[0] = y_[0].to(device)
                y_[1] = y_[1].to(device)
                y_[2] = y_[2].to(device)
                yy = model(x_[0],x_[1],x_[2])

                yy0 = yy[0]#.cpu().detach()
                yy1 = yy[1]#.cpu().detach()

                vl = loss_s(yy[0],y_[0]) + loss_e(yy[1],y_[1]) / 2
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

    torch.save(model.state_dict(), 'span_model')