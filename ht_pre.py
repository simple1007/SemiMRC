import pandas as pd
from tokenization_kobert import KoBertTokenizer

tokenizer = KoBertTokenizer.from_pretrained('monologg/distilkobert')

def fileread(file):
    return pd.read_csv(file)

def token(x):
    x = x.split()
    
    result = []
    for xx in x:
        xx = xx.split('+')
        temp = []
        for xxx in xx:
            temp.append(xxx.split('/')[0])
        temp = '+'.join(temp)
        result.append(temp)
    
    return ' '.join(result)

def only_ht():
    df = fileread('ori.csv')
    pddf = None
    pddf_err = None
    pddf_norm = None

    x_data = df.loc[:,['context_ht','question_ht','label_ht']]
    x_data_norm = df.loc[:,['context','question','text','start','end']]
    count = 0
    index = 0
    index_err = 0
    for rownum,t in enumerate(x_data.iterrows()):
        # print(t)
        # print(t[0])
        label = t[1]['label_ht']
        context = t[1]['context_ht']
        question = t[1]['question_ht']

        label = token(label)
        context = token(context)
        question = token(question)
        # print(label,context,question)
        # exit()
        data_norm = x_data_norm.iloc[rownum:rownum+1]#[rownum]
        args = {'context':data_norm['context'].item(),'question':data_norm['question'].item(),'text':data_norm['text'].item(),'start':data_norm['start'].item(),'end':data_norm['end'].item()}
        flag = ' '+label+' ' in context
        flag2 = ' '+label+'+' in context
        if flag or flag2:
           
            if flag2:
                l_i = context.index(' '+label+'+')+1
                # print(l_i,l_i+len(label),label,context[l_i:l_i+len(label)])

            elif flag:#flag:
                l_i = context.index(' '+label+' ')+1
            
            # print(label,context[l_i:l_i+len(label)])
            start = l_i
            end = l_i+len(label)
            
            count += 1

            data,data_err,data_norm_ = preprocessing(context,question,start,end,label,args)
            # print(data)
            # if data:# or len(data) != 0:
            if index == 0:
                pddf = pd.DataFrame(data)
                pddf_norm = pd.DataFrame(data_norm_)
                # pddf_err = pd.DataFrame(data_err)
                index+=1
            else:
                pddf2 = pd.DataFrame(data)
                pddf = pd.concat([pddf,pddf2],axis=0)
                index+=1

                pddf2_norm = pd.DataFrame(data_norm_)
                pddf_norm = pd.concat([pddf_norm,pddf2_norm],axis=0)
                # pddf2_err = pd.DataFrame(data_err)
                # pddf_err = pd.concat([pddf_err,pddf2_err])
            
            # if data_err:# or len(data_err) != 0:
            if index_err == 0:
                # pddf = pd.DataFrame(data_)
                pddf_err = pd.DataFrame(data_err)
                index_err+=1
                if pddf_err.shape[0] > 0:
                    print(rownum)
            else:
                # pddf2 = pd.DataFrame(data)
                # pddf = pd.concat([pddf,pddf2],axis=0)

                pddf2_err = pd.DataFrame(data_err)
                pddf_err = pd.concat([pddf_err,pddf2_err])
                if pddf2_err.shape[0] > 0:
                    print(rownum)
        else:
            False
        
        # if index == 100:
        #     print(data)
        #     break
            #print(label)#,t[1]['context_ht'])
    pddf.to_csv('{}.csv'.format('ht'),index=False,encoding=enc)
    pddf_norm.to_csv('{}.csv'.format('no_ht'),index=False,encoding=enc)
    #if pddf_err != None:
    pddf_err.to_csv('{}.csv'.format('ht_err'),index=False,encoding=enc)
    # print(t[1]['context_ht'])
    # print(len(x_data))
    # print(count)
    # print(len(x_data)-count)
lents = []
lente = []
enc = 'utf-8'
maxlen = 509#420
q_len = 63
c_len = 354
q_len = 99
c_len = 410
stop_ = ['\n','\r','\t','\v','\b']#,'\"',"\'"]
def preprocessing(context,question,s,e,label,args,span_flag=False):
    global lente, lents
    # doc_ = json_['paragraphs']
    data = []
    data_err = []
    ct_,q,cht,qht,flags,s_,e_,texts_ = [[] for _ in range(8)]
    s_l = []
    e_l = []
    t_l = []
    data_norm = []
    # for doc in doc_:
    context__ = context#doc['context']

    # for d in doc['qas']:#[1:]:
    ques = question#d['question']

    sss = ''; eee = ''; txt = '';
    if not span_flag:
        start_ = s#d['answers']['answer_start']
        end_ = e#d['answers']['answer_start'] + len(d['answers']['text'])
        text_ = label#d['answers']['text']
        sss = start_
        eee = end_
        txt = text_
    # else:
    #     start_ = d['answers']['clue_start']
    #     end_ = d['answers']['clue_start'] + len(d['answers']['clue_text'])
    #     text_ = d['answers']['clue_text']

    temp_c = []
    
    start__ = start_
    end__ = end_
    for c_index, c in enumerate(context__):
        if c not in stop_:
            temp_c.append(c)
        else:
            if c_index < start__:
                start_ -= 1
            if c_index < end__ :
                end_ -= 1
    ttt = ''.join(temp_c)[start_:end_]
    tt = text_
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
    
    sy = start_
    ey = end_

    start = 0
    end = 0
    ll = []

    count = 0
    flag = False
    for index, xx in enumerate(len_context):
        for xxx in xx:
            if (count-1) == sy:
                start = index
            elif (count-1) == ey:
                end = index
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
        # continue
        return None
    if end >= maxlen:
        # continue
        return None
    if flag:
        lents.append(start)
        lente.append(end)
        new_s = end + 1
        new_e = end + end - start + 1
        
        new_text = ''.join(len_context[start:end])
        
        while False:
            temp_text = ''.join(len_context[new_s:new_e])
            if new_text == temp_text:
                data.append({'flag':True,'context':tokenizer.convert_tokens_to_string(len_context),'question':tokenizer.convert_tokens_to_string(len_ques),'start':new_s+q_,'end':new_e+q_,'text':''.join(len_context[new_s:new_e])})
            new_s += 1
            new_e += 1
            data_norm.append(args)
            if new_s >= maxlen or new_s >= len(len_context) or new_e >= maxlen or new_e >= len(len_context):
                break
        if txt != tokenizer.convert_tokens_to_string(len_context[start:end]):
            # print(txt,len_context[start:end])
            data_err.append({'s_l':sss,'e_l':eee,'t_l':txt,'flag':False,'context':tokenizer.convert_tokens_to_string(len_context),'question':tokenizer.convert_tokens_to_string(len_ques),'start':start+q_,'end':end+q_,'text':''.join(len_context[start:end])})#,'morphs':context_morphs,'pos':context_pos})
            
        if tt == ttt:
                # exit()
            # else:
            data.append({'s_l':sss,'e_l':eee,'t_l':txt,'flag':False,'context':tokenizer.convert_tokens_to_string(len_context),'question':tokenizer.convert_tokens_to_string(len_ques),'start':start+q_,'end':end+q_,'text':''.join(len_context[start:end])})#,'morphs':context_morphs,'pos':context_pos})
            data_norm.append(args)
    return data,data_err,data_norm
if __name__ == '__main__':
    only_ht()
