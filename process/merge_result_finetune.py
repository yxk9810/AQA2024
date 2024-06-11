#coding:utf-8
import sys
import json

# 这里的名字与脚本不太一致，是因为脚本重新把代码优化过了

top_n =30
e5_mis = []
with open('../preds/e5_7b_top50.txt','r',encoding='utf-8') as lines:
    for line in lines:
        data = line.strip().split(',')[:top_n]
        new_dic = {d.split(' ')[0]:float(d.split(' ')[1]) for d in data}
        e5_mis.append(new_dic)

#
linq_lora = []
with open('../preds/linq_7b_top50.txt','r',encoding='utf-8') as lines:
    for line in lines:
        data = line.strip().split(',')[:top_n]
        new_dic = {d.split(' ')[0]:float(d.split(' ')[1]) for d in data}
        linq_lora.append(new_dic)


sfr_lora = []
with open('../preds/sfr_7b_top50.txt','r',encoding='utf-8') as lines:
    for line in lines:
        data = line.strip().split(',')[:top_n]
        new_dic = {d.split(' ')[0]:float(d.split(' ')[1]) for d in data}
        sfr_lora.append(new_dic)



gte_lora = []
with open('../preds/gte_7b_top50.txt','r',encoding='utf-8') as lines:
    for line in lines:
        data = line.strip().split(',')[:top_n]
        new_dic = {d.split(' ')[0]:float(d.split(' ')[1]) for d in data}
        gte_lora.append(new_dic)

liq_re = []
with open('../preds/Linq-Embed-Mistral_result.txt','r',encoding='utf-8') as lines:
    for line in lines:
        data = line.strip().split(',')[:top_n]
        new_dic = {d.split(' ')[0]:float(d.split(' ')[1]) for d in data}
        liq_re.append(new_dic)



nv_re = []
with open('../preds/NV-Embed-v1_result.txt','r',encoding='utf-8') as lines:
    for line in lines:
        data = line.strip().split(',')[:top_n]
        new_dic = {d.split(' ')[0]:float(d.split(' ')[1]) for d in data}
        nv_re.append(new_dic)


writer = open('merge_lora_4_top30.txt','a+',encoding='utf-8')
for t_s,t_l,t_n,lq_lora,sf_lora,gt_lora in zip(e5_mis,liq_re,nv_re,linq_lora,sfr_lora,gte_lora):
    new_scores = {}
    pids = set(t_s.keys()).union(set(t_l.keys())).union(set(t_n.keys())).union(set(lq_lora.keys())).union(set(sf_lora.keys())).union(set(gt_lora.keys()))
    for id in pids:
        score = [0.0] * 6
        if id in t_s:
            score[0] = t_s[id]
        if id in t_l:
            score[1] = t_l[id]
        if id in t_n:
            score[2] = t_n[id]
        if id in lq_lora:
            score[3] = lq_lora[id]
        if id in sf_lora:
            score[4] = sf_lora[id]
        if id in gt_lora:
            score[5] = gt_lora[id]
        # 所有
        t_score = score[0]+score[3]+score[4]+score[1]+score[2]+score[5]
        # 所有lora结果
        # t_score = score[0]+score[3]+score[4]+score[5]
        new_scores[id] = t_score
    top20 = sorted(new_scores.items(), key=lambda x: x[1], reverse=True)[:20]
    # print(top20)
    writer.write(','.join([w[0] for w in top20][:20]) + '\n')
writer.close()