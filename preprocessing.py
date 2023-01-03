#-*coding: utf-8 -*-
import json
import os
import string
import pandas as pd
from tokenization_kobert import KoBertTokenizer
from konlpy.tag import Okt

okt = Okt()
tokenizer = KoBertTokenizer.from_pretrained('monologg/distilkobert')
enc = 'utf-8'
maxlen = 420
q_len = 63
c_len = 354
def load_dataset(file):
    with open(file,encoding=enc) as f:
        rep_char = '-'
        text_ = f.read()#.replace('ㆍ',rep_char).replace('\\n',rep_char).replace('\\r',rep_char).replace('\\t',rep_char).replace('\\v',rep_char).replace('\\b',rep_char).replace('\\"',rep_char).replace("\\'",rep_char)
        # v = ['"','{','}',':','[',']',',']
        # for p in string.punctuation:
            # if p not in v:
            # text_ = text_.replace(p,'▁')
        return json.loads(text_)
lents = []
lente = []
# stop_ = ['ㆍ',
stop_ = ['\n','\r','\t','\v','\b']#,'\"',"\'"]

def preprocessing(json_,span_flag):
    global lente, lents
    doc_ = json_['paragraphs']
    data = []
    
    for doc in doc_:
        context__ = doc['context']

        for d in doc['qas']:#[1:]:
            ques = d['question']
            if not span_flag:
                start_ = d['answers']['answer_start']
                end_ = d['answers']['answer_start'] + len(d['answers']['text'])
                text_ = d['answers']['text']
            else:
                # print(flag)
                # print(d['answers'])
                start_ = d['answers']['clue_start']
                end_ = d['answers']['clue_start'] + len(d['answers']['clue_text'])
                text_ = d['answers']['clue_text']

            temp_c = []
            
            # start = d['answers']['answer_start']
            start__ = start_
            # end = start + len(d['answers']['text'])
            end__ = end_
            for c_index, c in enumerate(context__):
                if c not in stop_:
                    temp_c.append(c)
                else:
                    # if c == stop_[0]:
                    #     temp_c.append('_')
                    #     continue
                    if c_index < start__:
                        start_ -= 1
                    if c_index < end__ :
                        end_ -= 1
            ttt = ''.join(temp_c)[start_:end_]
            tt = text_#doc['qas'][0]['answers']['text']
            # print(json_['doc_id'],ttt,tt,ttt==tt)
            context_ = ''.join(temp_c)
            x = context_

            x = x.replace('㏊','-')
            f = x
            
            len_ques = tokenizer.tokenize(ques)
            len_ques = len_ques[:q_len]
            len_context = tokenizer.tokenize(context_)
            len_context = len_context[:c_len]

            q_ = len(len_ques)
            c_ = len(len_context)
            # print(xx)
            # print(xx)
            
            sy = start_
            ey = end_

            start = 0
            end = 0
            ll = []
            # ff = tokenizer.tokenize(c)
            count = 0
            flag = False
            for index, xx in enumerate(len_context):
                for xxx in xx:
                    if (count-1) == sy:
                        # print(4,count,sy)
                        start = index
                    elif (count-1) == ey:
                        end = index
                        # print(5,count,ey)
                        flag = not flag
                        break
                    count += 1
                if flag:
                    break
            ff_ = ''.join(len_context[start:end])
            if ff_.startswith('▁'):
                ff_ = ff_[1:]
            if len(ff_) < (ey - sy):
                end += 1

            if start >= maxlen:
                continue
            if end >= maxlen:
                continue
            if flag:
                lents.append(start)
                lente.append(end)
                new_s = end + 1
                new_e = end + end - start + 1
                
                new_text = ''.join(len_context[start:end])
                # new_temp = []
                while False:
                    temp_text = ''.join(len_context[new_s:new_e])
                    if new_text == temp_text:
                        data.append({'flag':True,'context':tokenizer.convert_tokens_to_string(len_context),'question':tokenizer.convert_tokens_to_string(len_ques),'start':new_s+q_,'end':new_e+q_,'text':''.join(len_context[new_s:new_e])})
                    new_s += 1
                    new_e += 1

                    if new_s >= maxlen or new_s >= len(len_context) or new_e >= maxlen or new_e >= len(len_context):
                        break 
                if tt == ttt:
                    data.append({'flag':False,'context':tokenizer.convert_tokens_to_string(len_context),'question':tokenizer.convert_tokens_to_string(len_ques),'start':start+q_,'end':end+q_,'text':''.join(len_context[start:end])})#,'morphs':context_morphs,'pos':context_pos})
    return data

def to_dataframe(jsons_,filename,span_clue):
    df = None
    count = 0
    # print(span_clue)
    from tqdm import tqdm
    total = len(jsons_[0]['data']) + len(jsons_[1]['data'])
    with tqdm(total = total) as pbar:
        for jndex, json_ in enumerate(jsons_):
            flag = span_clue[jndex]
            # print(jndex)
            for index, j in enumerate(json_['data']):
                data = preprocessing(j,flag)
                count += len(data)
                # continue
                if index == 0 and jndex == 0:
                    df = pd.DataFrame(data)
                else:
                    df2 = pd.DataFrame(data)
                    df = pd.concat([df,df2],axis=0)
                if flag:
                #     for index, j in enumerate(json_['data']):
                    data = preprocessing(j,False)
                    count += len(data)

                    df2 = pd.DataFrame(data)
                    df = pd.concat([df,df2],axis=0)
                pbar.update(1)
        print(count)
        # df = df.sample(frac=1).reset_index(drop=True)
        df.to_csv('{}.csv'.format(filename),index=False,encoding=enc)

# load_dataset('TL_unanswerable.json')
files = os.listdir('.')
datas = []
span_clue = []
for file in files:
    if file.endswith('.json'):
        if file != 'TL_span_extraction.json' and file != 'TL_span_inference.json':
            continue
        print(file)
        flag = False
        if 'inference' in file:
            flag = True
        flag=False
        span_clue.append(flag)
        jsons = load_dataset(file)
        datas.append(jsons)
# print(span_clue)
to_dataframe(datas,'data',span_clue)
# print('s',max(lents),'e',max(lente))

exit()
file = 'TL_span_extraction.json'
json1 = load_dataset(file)
with open(file,encoding=enc) as f:
    json2 = json.loads(f.read())
# jsons2 = load_dataset(file)
# for json1_ ,json2_ in zip(json1,json2):
#     print(json1_)
cnt = 0
for j1, j2 in zip(json1['data'],json2['data']):
    # json1_ = json1['data']
    # json2_ = json2['data']
    
    doc1_ = j1['paragraphs']
    doc2_ = j2['paragraphs']
    # data = []
    for doc1,doc2 in zip(doc1_,doc2_):
        
        # print(doc['context'])
        # doc['context']
        # doc['qas'][0]['question']
        # doc['qas'][0]['answers']['answer_start']
        # doc['qas'][0]['answers']['answer_start']+len(doc['qas'][0]['answers']['text'])

        # data.append({'context':doc['context'],'question':doc['qas'][0]['question'],'start':doc['qas'][0]['answers']['answer_start'],'end':doc['qas'][0]['answers']['answer_start']+len(doc['qas'][0]['answers']['text']),'text':doc['qas'][0]['answers']['text']})
        context1_ = doc1['context']
        context2_ = doc2['context']
        if len(context1_) != len(context2_):
            print('c',doc1['doc_id'])
            cnt += 1
        # 320
        for d1,d2 in zip(doc1['qas'],doc2['qas']):#[1:]:
            # ques = d['question']
            if len(d1['question']) != len(d2['question']):
                print('q',1)
            # start_ = d['answers']['answer_start']
            # end_ = d['answers']['answer_start'] + len(d['answers']['text'])
            # text_ = d['answers']['text']

print(cnt)