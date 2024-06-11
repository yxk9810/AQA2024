#coding:utf-8
import sys 
import json
import re
def fielter_url(text):
    text = text.replace("\n", "")
    text = re.sub(r'<.*?>', "", text)
    text = " ".join([x for x in text.strip().split(" ") if not ("<" in x or "/" in x or ">" in x)])
    return text

#process test query file 
data_dir = 'AQA-test-public/'
qa_test_file = data_dir+'/'+'qa_test_wo_ans_new.txt'
dest_dir = './'
writer = open(dest_dir+'test_data_0606.tsv','a+',encoding='utf-8')
with open(qa_test_file, "r") as lines:
    for idx,line in enumerate(lines):
        data = json.loads(line.strip())
        query =data['query']
        body = fielter_url(data['body']) if 'body' in data else ''
        writer.write(json.dumps({'query':query+body,'query_id':idx+1})+'\n')
writer.close()


#process corpus data 

corpus_file = 'pid_to_title_abs_update_filter.json'
from tqdm import tqdm 
with open(data_dir+corpus_file) as rf:
    pid2info = json.load(rf)
writer = open(dest_dir+'test_corpus_data.jsonl','a+',encoding='utf-8')
for i, pid in tqdm(enumerate(pid2info)):
    info = pid2info[pid]
    json_data = {'title':info['title'] if 'title' in info and info['title'] else '',
    'text':info['abstract'],'docid':pid}
    writer.write(json.dumps(json_data,ensure_ascii=False)+'\n')
writer.close()