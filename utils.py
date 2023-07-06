import pandas as pd
import torch
import re
from tokenization_kobert import KoBertTokenizer
# TL_span_extraction.csv
maxlen = 509#420
def fileread(file):
    return pd.read_csv(file)

def get_v(file):
    reader = fileread(file)
    d = reader.loc[:,['start','end']]
    # print(reader['question'])
    a = []
    for dd in d['end']:
        a.append(dd)

    print(max(a))
    
    x_data = reader.loc[:,['context','question']]

class MRCDataset(torch.utils.data.Dataset): 
  def __init__(self,reader,w2i,ht_flag=False,normal_tk=False):
    self.tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
    self.reader = reader
    # print('length:',self.reader.shape[0])
    if ht_flag:
      self.x_data = reader.loc[:,['context','question','context_ht','question_ht']]
    else:
      self.x_data = reader.loc[:,['context','question']]
    self.y_data = reader.loc[:,['start','end']]
    self.w2i = w2i
    self.ht_flag = ht_flag
    self.normal_tk = normal_tk

  def get_noun(self,text):
    text = text.split()
    result = []
    for t in text:
      # for tt in t.split('')
      t = t.split('+')[0]
      #if '/S' not in t:
      if '/N' in t:
        #result.append(t)
        result.append(t.split('/')[0])
    # print(result)
    return ' '.join(result)
  
  def len_w2i(self):
    return len(self.w2i.keys())
  
  def __len__(self):
    # return 106
    return self.reader.shape[0]

  def __getitem__(self, idx):
    x1 = self.x_data.iloc[idx:idx+1]['context'].item() #+ ' [SEP] ' + 
    x2 = self.x_data.iloc[idx:idx+1]['question'].item()
    if self.normal_tk:
        x1 = x1.replace('+','')
        x2 = x2.replace('+','')
        
        x1 = re.sub('([0-9+]) ',r'\1',x1)
        x2 = re.sub('([0-9+]) ',r'\1',x2)
        #print(x2,x1)
    x = self.tokenizer(x2,x1,padding="max_length", max_length=maxlen, truncation=True)
    
    if self.ht_flag:
    	
      htc = self.x_data.iloc[idx:idx+1]['context_ht'].item()
      htq = self.x_data.iloc[idx:idx+1]['question_ht'].item()
      
      

      htc = self.get_noun(htc)
      htq = self.get_noun(htq)

      htqc = htq.split() + htc.split()
      # print(len(htqc))
      ht = []
      # print(htqc)
      for m in htqc:
        if m in self.w2i:
          ht.append(self.w2i[m])
        else:
          ht.append(self.w2i['[UNK]'])
      # print(ht)
      ht = ht[:maxlen] + [self.w2i['[PAD]']] * (maxlen - len(ht[:maxlen]))
    # ht = self.tokenizer(htq,htc,padding="max_length", max_length=maxlen, truncation=True)

    sy = self.y_data.iloc[idx:idx+1]['start'].item() + 2
    ey = self.y_data.iloc[idx:idx+1]['end'].item() + 2
    
    seq_s = sy
    seq_e = ey
    label = [0] * maxlen

    for i in range(seq_s,seq_e):
      label[i] = 1
    
    # print(torch.tensor(ht).shape)
    # print(torch.tensor(x['token_type_ids'],dtype=torch.long).shape)
    if self.ht_flag:
      return [torch.tensor(x['input_ids'],dtype=torch.long),torch.tensor(x['token_type_ids'],dtype=torch.long),torch.tensor(x['attention_mask'],dtype=torch.long)], torch.tensor(ht,dtype=torch.long),[torch.tensor(seq_s,dtype=torch.long),torch.tensor(seq_e,dtype=torch.long),torch.tensor(label)]#, [f[sy:ey]]#[torch.tensor(sy, dtype=torch.long),torch.tensor(ey, dtype=torch.long)]
    else:
      return [torch.tensor(x['input_ids'],dtype=torch.long),torch.tensor(x['token_type_ids'],dtype=torch.long),torch.tensor(x['attention_mask'],dtype=torch.long)], torch.tensor([1],dtype=torch.long),[torch.tensor(seq_s,dtype=torch.long),torch.tensor(seq_e,dtype=torch.long),torch.tensor(label)]#, [f[sy:ey]]#[torch.tensor(sy, dtype=torch.long),torch.tensor(ey, dtype=torch.long)]
def get_noun(text):
  text = text.split()
  result = []
  for t in text:
    # for tt in t.split('')
    t = t.split('+')[0]
    #if '/S' not in t:
    if '/N' in t:
      #result.append(t)
      result.append(t.split('/')[0])#.split('/')[0])
  # print(result)
  return ' '.join(result)

def preprocess_w2i(reader):
  # data = reader.loc[:,['context','question','context_ht','question_ht']]
  w2i = {}  
  w2i['[PAD]'] = 0
  w2i['[UNK]'] = 1
  index = 2
  for i,d in reader.iterrows():
    # print(d)
    c = get_noun(d['context_ht'])
    q = get_noun(d['question_ht'])

    cq = c.split() + q.split()

    for w in cq:
      if w not in w2i:
        w2i[w] = index
        index += 1
    # print(c,q)

  import pickle
  with open('noun_model/w2i.pkl','wb') as f:
    pickle.dump(w2i,f)     

if __name__ == '__main__':
    filename = 'data.csv'
    import pickle
    #with open('noun_model/w2i.pkl','rb') as f:
    #  w2i = pickle.load(f)     

    reader = fileread(filename)
    preprocess_w2i(reader)
    exit()
    mrc_data = MRCDataset(reader,w2i)

    dataloader = torch.utils.data.DataLoader(dataset=mrc_data,
                        batch_size=1,
                        shuffle=True,
                        drop_last=False)
    for batch in dataloader:
        x,ht,y = batch
        print(ht.shape)
        # print(ht)
        exit()        
