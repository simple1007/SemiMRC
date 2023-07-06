import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from transformers import BertModel, DistilBertModel
from utils import MRCDataset, fileread
from train import BERTFineTune, BERTHTFineTune
from torch.utils.data import random_split
from tqdm import tqdm
import pickle
with open('ht_model/w2i.pkl','rb') as f:
    w2i = pickle.load(f)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
reader = fileread('no_ht.csv')

prefix = "no_model"
model_file = prefix+"/"+prefix

mrc_data = MRCDataset(reader,w2i,False,False)
batch_size = 100
dataloader = torch.utils.data.DataLoader(dataset=mrc_data,
    batch_size=100,
    # shuffle=True,
    drop_last=False)

# train_size = int(len(mrc_data) * 0.8)
# validation_size = int(len(mrc_data)*0.02)# - train_size
test_size = 100000
test = (len(mrc_data)//batch_size) - (test_size//batch_size)#test_size - validation_size - train_size
# test_size = len(mrc_data) - train_size - validation_size
# train_dataset, validation_dataset,test_dataset, _ = random_split(mrc_data, [train_size, validation_size,test_size,test])
# testloader = torch.utils.data.DataLoader(dataset=test_dataset,
#     batch_size=batch_size,
#     # shuffle=False,
#     drop_last=True) 


bert_model = DistilBertModel.from_pretrained('monologg/distilkobert', return_dict=False)

# model = BERTHTFineTune(bert_model,mrc_data.len_w2i())
model = BERTFineTune(bert_model)

model.load_state_dict(torch.load(model_file))

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
    total_pre_val = 0.0
    total_rec_val = 0.0
    s1 = 0.0
    s2 = 0.0
    count = 0
    log = open('res.txt','w',encoding='utf-8')
    for index_, batch in enumerate(tqdm(dataloader)):
        if index_ < (test // batch_size):
            continue
        x, ht,y = batch
        # print(ht)
        x[0] = x[0].to(device)
        x[1] = x[1].to(device)
        x[2] = x[2].to(device)
        y[0] = y[0].to(device)
        y[1] = y[1].to(device)
        ht=ht.to(device)
        sy, ey = model(x[0],x[1],x[2])#,ht)
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
            
            re = re.strip('+')
            if re != re2:
                # if re.strip() == '':
                #     continue
                temp = defaultdict(int)
                re_ = re.split()
                re2_ = re2.split()
                val = 0
                for r in re_:
                    temp[r] += 1
                for r2 in re2_:
                    if r2 in temp:
                        val += 1
                # print(re,re_,re2,re2_)
                # for i in range(len(re2)-1):
                #     temp[re2[i:i+2]] += 1
                # if len(re2) == 1:
                #     temp[re2] += 1
                # predict_bi = 0
                # temp_tr = defaultdict(int)
                # for i in range(len(re)-1):
                #     temp_tr[re[i:i+2]] += 1
                #     if re[i:i+2] in temp:
                #         predict_bi += 1
                # if len(re) == 1:
                #     temp_tr[re] += 1
                # print('aaa'+re+'bbb')
                
                # print(re,',,',re2,',,',temp,sum(temp.values()))
                pre_val = val / (len(re_)+1e-9)#(sum(temp_tr.values())+1e-9)#sum(temp.values())
                rec_val = val / (len(re2_)+1e-9)#(sum(temp.values())+1e-9)
                total_pre_val += pre_val
                total_rec_val += rec_val
                if False:#pre_val > 1.0 or pre_val >= 0.5:
                    result2+=1
                    log.write(str(pre_val)+':'+re+':'+re2+'\n')
                log.write("correct: "+re2+", predict: "+re+"\n")
                f.write(str(pre_val)+'\t'+str(index)+'\t'+str(re)+'\t'+str(re2)+'\n')
            if re == re2:
                result += 1
                result2 += 1
                total_pre_val += 1.0
                total_rec_val += 1.0

        s = torch.eq(start,y[0])#.shape[0]
        e = torch.eq(end,y[1])#.shape[0]

        scount += s[s==True].shape[0]
        ecount += e[e==True].shape[0]
        if count >= test_size:
            break
    print(prefix,"rec_val",total_rec_val/count,"pre_val",total_pre_val/count,"re2",result2/count,"re",result/count,'s',scount/count,'cs_loss',s1/(count//batch_size),'e',ecount/count,'ce_loss',s2/(count//batch_size))
    recall = total_rec_val/count
    precision = total_pre_val/count
    print("f1 score: ",((2*precision*recall) / (precision+recall)))
f.close()
log.close()
        # exit()
