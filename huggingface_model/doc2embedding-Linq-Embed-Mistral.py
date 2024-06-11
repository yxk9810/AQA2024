import csv
from tqdm import tqdm
import os
import json
import transformers
import torch.nn.functional as F
transformers.logging.set_verbosity_error()
import pdb
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    BertTokenizer,
    BertModel,
    )
from transformers import AutoTokenizer,AutoModel
import torch
import numpy as np
from accelerate import PartialState
import torch
import torch.nn.functional as F
from torch import Tensor


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

@torch.no_grad()
def encode(model,texts,tokenizer,device):
    batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors="pt").to(device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return embeddings
    

task = 'Given a question, retrieve Wikipedia passages that answer the question'

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--wikipedia_path",default="data/pid_to_title_abs_update_filter.json")
    parser.add_argument("--name",default='large')
    parser.add_argument("--encoding_batch_size",type=int,default=1)
    parser.add_argument("--pretrained_model_path",default="nvidia")
    parser.add_argument("--output_dir",default="./")
    args = parser.parse_args()
    args.pretrained_model_path = "Linq-AI-Research/Linq-Embed-Mistral"

    distributed_state = PartialState()
    device = distributed_state.device

    ## load encoder
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path)
    doc_encoder = AutoModel.from_pretrained(args.pretrained_model_path,trust_remote_code=True)
    doc_encoder.eval()
    doc_encoder.to(device)

    wikipedia = []
    data = json.load(open("data/pid_to_title_abs_update_filter.json","r",encoding="utf-8"))
    for k in tqdm(data.keys()):
        title = data[k]["title"]
        abstract = data[k]["abstract"]
        title = title.strip() if title else " "
        abstract = abstract.strip() if abstract else " "
        wikipedia.append(title+","+abstract)

    print("candidates papers:{}".format(len(wikipedia)))
    passage_prefix = ""
    with distributed_state.split_between_processes(wikipedia) as sharded_wikipedia:
        print(len(sharded_wikipedia))
        sharded_wikipedia = [sharded_wikipedia[idx:idx+args.encoding_batch_size] for idx in range(0,len(sharded_wikipedia),args.encoding_batch_size)]
        encoding_progress_bar = tqdm(total=len(sharded_wikipedia), disable=not distributed_state.is_main_process,ncols=100,desc='encoding wikipedia...')
        doc_embeddings = []
        for data in sharded_wikipedia:
            passage_embeddings = encode(doc_encoder,data,tokenizer,device)
            passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)
            passage_embeddings = passage_embeddings.cpu().numpy()
            doc_embeddings.append(passage_embeddings)
            encoding_progress_bar.update(1)
        doc_embeddings = np.concatenate(doc_embeddings,axis=0)

        np.save('papers_emb/Linq-Embed-Mistral/doc_embeddings_{}.npy'.format(distributed_state.process_index),doc_embeddings)

