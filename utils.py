import pandas as pd
import torch

from tokenization_kobert import KoBertTokenizer
# TL_span_extraction.csv
maxlen = 420
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
  def __init__(self,reader):
    self.tokenizer = KoBertTokenizer.from_pretrained('monologg/distilkobert')
    self.reader = reader
    print('length:',self.reader.shape[0])
    self.x_data = reader.loc[:,['context','question']]
    self.y_data = reader.loc[:,['start','end']]

  def __len__(self):
    # return 10
    return self.reader.shape[0]

  def __getitem__(self, idx):
    x1 = self.x_data.iloc[idx:idx+1]['context'].item() #+ ' [SEP] ' + 
    x2 = self.x_data.iloc[idx:idx+1]['question'].item()

    x = self.tokenizer(x2,x1,padding="max_length", max_length=maxlen, truncation=True)
    
    sy = self.y_data.iloc[idx:idx+1]['start'].item() + 2
    ey = self.y_data.iloc[idx:idx+1]['end'].item() + 2
    
    seq_s = sy
    seq_e = ey
    label = [0] * maxlen

    for i in range(seq_s,seq_e):
      label[i] = 1
    return [torch.tensor(x['input_ids'],dtype=torch.long),torch.tensor(x['token_type_ids'],dtype=torch.long),torch.tensor(x['attention_mask'],dtype=torch.long)], [torch.tensor(seq_s,dtype=torch.long),torch.tensor(seq_e,dtype=torch.long),torch.tensor(label)]#, [f[sy:ey]]#[torch.tensor(sy, dtype=torch.long),torch.tensor(ey, dtype=torch.long)]

if __name__ == '__main__':
    filename = 'data.csv'

    reader = fileread(filename)
    
    mrc_data = MRCDataset(reader)

    dataloader = torch.utils.data.DataLoader(dataset=mrc_data,
                        batch_size=1,
                        shuffle=True,
                        drop_last=False)
    for batch in dataloader:
        x,y = batch
        