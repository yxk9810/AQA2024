#coding:utf-8
import sys
import json

dic = {}
with open('question2body.json','r',encoding='utf-8') as lines:
    for line in lines:
        data = json.loads(line.strip())
        dic[data['question']] = data['body']

writer = open('/mnt/workspace/data/AIME/index/wjd/tevatron/aqa_train_data_processed/test_data_0606.tsv','a+',encoding='utf-8')
with open('/mnt/workspace/data/AIME/index/wjd/tevatron/aqa_train_data_processed/test_data_0603.tsv','r',encoding='utf-8') as lines:
    for line in lines:
        data = json.loads(line.strip())
        query =data['query']
        content = data['query']+dic.get(query,'')
        data['query'] = content
        writer.write(json.dumps(data,ensure_ascii=False)+'\n')
writer.close()