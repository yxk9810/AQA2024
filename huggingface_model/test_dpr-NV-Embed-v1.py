from utils import normalize_query
import csv
import faiss,pickle        
import numpy as np 
from tqdm import tqdm
from transformers import DPRQuestionEncoder,DPRQuestionEncoderTokenizer,BertModel,BertTokenizer
import torch
from utils.tokenizers import SimpleTokenizer
import unicodedata
import time
import pdb
import torch.nn.functional as F
import transformers
import json
transformers.logging.set_verbosity_error()
from transformers import AutoTokenizer, AutoModel
    
question2body = [json.loads(line.strip()) for line in open("data/question2body.json","r",encoding="utf-8")]
question2body = {x["question"]:x["body"] for x in question2body}

def normalize(text):
    return unicodedata.normalize("NFD", text)

def has_answer(answers,doc):
    tokenizer = SimpleTokenizer()
    doc = tokenizer.tokenize(normalize(doc)).words(uncased=True)
    for answer in answers:
        answer = tokenizer.tokenize(normalize(answer)).words(uncased=True)
        for i in range(0, len(doc) - len(answer) + 1):
                if answer == doc[i : i + len(answer)]:
                    return True
    return False
    
  
def gen_format(scores,ids):
    res = []
    assert len(scores)==len(ids)
    for x,y in zip(scores,ids):
        assert len(x)==len(y)
        tmp = [(x,y) for x,y in zip(x,y)]
        res.append(tmp)
    return res

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nq_test_file",default="data/qa_test_wo_ans_new.txt")
    parser.add_argument("--num_shards",type=int,default=1)
    parser.add_argument("--pretrained_model_path",default="nvidia")
    parser.add_argument("--name",default='large')
    args = parser.parse_args()
    args.pretrained_model_path = "nvidia/NV-Embed-v1"

    ## load QA dataset
    queries = []
    with open(args.nq_test_file) as f:
        for line in f:
            q = json.loads(line.strip())["question"]
            body = question2body.get(q," ")
            queries.append((normalize_query(q),normalize_query(body)))

    
    # make faiss index
    embedding_dimension = 4096 
    index = faiss.IndexFlatIP(embedding_dimension)
    for idx in tqdm(range(args.num_shards),desc='building index from embedding...'):
        data = np.load("papers_emb/NV-Embed-v1/doc_embeddings_{}.npy".format(idx))
        index.add(data)
    

    ## load wikipedia passages
    data = json.load(open("data/pid_to_title_abs_update_filter.json","r",encoding="utf-8"))
    wikipedia_id = list(data.keys())

    ## load query encoder

    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    query_encoder = AutoModel.from_pretrained(args.pretrained_model_path,trust_remote_code=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_encoder.to(device).eval()
        
    task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}
    query_prefix = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "

    ## embed queries
    tmp,res = [],[]
    with open("../preds/NV-Embed-v1_result.txt","w",encoding="utf-8") as f:
        for query,body in tqdm(queries,desc='encoding queries...'):
            query_embedding = query_encoder.encode([query+","+body], instruction=query_prefix, max_length=4096)
            query_embedding = F.normalize(query_embedding, p=2, dim=1)
            query_embedding = query_embedding.cpu().detach().numpy()
            tmp.append(query_embedding)
            if len(tmp)==100:
                tmp = np.concatenate(tmp,axis=0)
                score,IS = index.search(tmp,50)
                IS = gen_format(score,IS)
                res+=[k for k in IS]
                tmp = []
        
        if len(tmp)>0:
            tmp = np.concatenate(tmp,axis=0)
            score,IS = index.search(tmp,50)
            IS = gen_format(score,IS)
            res+=[k for k in IS]
         
        for I in res:
            top_k = [wikipedia_id[x[1]]+" "+str(x[0]) for x in I]
            f.write("{}\n".format(",".join(top_k)))