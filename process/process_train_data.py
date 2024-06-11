import sys
import json
train_data_path = 'aqa_train_data_processed'
train_data = json.load(open(train_data_path+'train_with_hn.json','r',encoding='utf-8'))
writer = open('train_qwen_0507.jsonl','a+',encoding='utf-8')
for t_idx,d in enumerate(train_data):    
    new_json ={'query_id':str(t_idx+1),'query':d['question']}
    pos = d['positive_ctxs']
    for t_id,p in enumerate(pos):
        if 'title' in p and (not p['title'] or p['title'].strip()==''):
            p['title']=''
        p['doc_id'] = str(t_idx*10000+t_id)
    new_json['positive_passages'] = pos 
    neg = d['hard_negative_ctxs'] if len(d['hard_negative_ctxs'])>0 else d['negative_ctxs']
    for t_id,p in enumerate(neg):
        if 'title' in p and (not p['title'] or p['title'].strip()==''):
            p['title']=''
        p['doc_id'] = str(t_idx*20000+t_id)
    new_json['negative_passages'] = neg
    # print(new_json)
    writer.write(json.dumps(new_json,ensure_ascii=False)+'\n')
writer.close()
