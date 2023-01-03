import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from transformers import BertModel, DistilBertModel
from utils import MRCDataset, fileread
from train import BERTFineTune
from torch.utils.data import random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reader = fileread('data.csv')

mrc_data = MRCDataset(reader)

dataloader = torch.utils.data.DataLoader(dataset=mrc_data,
    batch_size=20,
    # shuffle=True,
    drop_last=False)

batch_size = 20
train_size = 50000 #int(len(mrc_data) * 0.8)
validation_size = 20000#len(mrc_data) - train_size
test_size = 20000
test = len(mrc_data) - test_size - validation_size - train_size
# test_size = len(mrc_data) - train_size - validation_size
train_dataset, validation_dataset,testloader, _ = random_split(mrc_data, [train_size, validation_size,test_size,test])

batch_size = 20

bert_model = DistilBertModel.from_pretrained('monologg/distilkobert', return_dict=False)

model = BERTFineTune(bert_model)

model.load_state_dict(torch.load('span_model'))

model.to(device)
model.eval()

loss_s = nn.NLLLoss()
loss_e = nn.NLLLoss()
f = open('result.txt','w',encoding='utf-8')
with torch.no_grad():
    scount = 0
    ecount = 0
    result = 0
    result2 = 0
    s1 = 0.0
    s2 = 0.0
    count = 0
    log = open('res.txt','w',encoding='utf-8')
    for index, batch in enumerate(dataloader):
        x, y = batch

        x[0] = x[0].to(device)
        x[1] = x[1].to(device)
        x[2] = x[2].to(device)
        y[0] = y[0].to(device)
        y[1] = y[1].to(device)
        sy, ey = model(x[0],x[1],x[2])
        ss = loss_s(sy,y[0]) + loss_s(ey,y[1])  /2#.item()
        s1 += ss.item()
        sy = sy#.detach() 
        ey = ey#.detach()

        start = torch.argmax(sy,dim=-1)
        end = torch.argmax(ey,dim=-1)
        
        for index,(s,e,ts,te,tx) in enumerate(zip(start,end,y[0],y[1],x[0])):
            count += 1
            s = s.cpu().detach().numpy()
            e = e.cpu().detach().numpy()
            # print(s,e,x[0])
            re = mrc_data.tokenizer.convert_ids_to_tokens(tx[s:e])
            
            ts = ts.cpu().detach().numpy()
            te = te.cpu().detach().numpy()
            # print(ts,te)
            re2 = mrc_data.tokenizer.convert_ids_to_tokens(tx[ts:te])

            re = mrc_data.tokenizer.convert_tokens_to_string(re)
            re2 = mrc_data.tokenizer.convert_tokens_to_string(re2)
            
            if re != re2:
                temp = defaultdict(int)
                for i in range(len(re2)-1):
                    temp[re2[i:i+2]] += 1
                if len(re2) == 1:
                    temp[re2] += 1
                predict_bi = 0
                for i in range(len(re)-1):
                    if re[i:i+2] in temp:
                        predict_bi += 1
                
                # print(re,',,',re2,',,',temp,sum(temp.values()))
                pre_val = predict_bi / sum(temp.values())
                if pre_val > 1.0 or pre_val >= 0.7:
                    result2+=1
                    log.write(str(pre_val)+':'+re+':'+re2+'\n')
                
                f.write(str(pre_val)+'\t'+str(index)+'\t'+str(re)+'\t'+str(re2)+'\n')
            if re == re2:
                result += 1
                result2 += 1

        s = torch.eq(start,y[0])#.shape[0]
        e = torch.eq(end,y[1])#.shape[0]

        scount += s[s==True].shape[0]
        ecount += e[e==True].shape[0]
     
    print("re2",result2/count,"re",result/validation_size,'s',scount/len(mrc_data),'cs_loss',s1/(len(mrc_data)//batch_size),'e',ecount/len(mrc_data),'ce_loss',s2/(len(mrc_data)//batch_size))
f.close()
log.close()
        # exit()