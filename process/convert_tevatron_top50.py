#coding:utf-8
import sys
import  json
import os
output_dir = './preds/'
for filename in os.listdir('./results'):
    if 'top50' not in filename:continue
    arr = []
    outfilename = filename.split('/')[-1]
    writer = open(output_dir+outfilename,'a+',encoding='utf-8')
    with open('./results/'+filename,'r',encoding='utf-8') as lines:
        for line in lines:
            data =line.strip().split('\t')
            if len(arr) ==50:
                t_list = []
                for d in arr:
                    t_list.append(d[1]+' '+d[-1])
                writer.write(','.join(t_list)+'\n')
                arr = []
            arr.append(data)
    if len(arr)>0:
        t_list = []
        for d in arr:
            t_list.append(d[1] + ' ' + d[-1])
        writer.write(','.join(t_list) + '\n')
    writer.close()