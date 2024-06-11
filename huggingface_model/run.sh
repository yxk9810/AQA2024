#!/bin/bash
python gen_query2body.py
# passage向量编码
accelerate launch --config_file default_config.yaml doc2embedding-NV-Embed-v1.py
# query 向量编码，检索top50
python test_dpr-NV-Embed-v1.py
# passage向量编码
accelerate launch --config_file default_config.yaml doc2embedding-Linq-Embed-Mistral.py
# query 向量编码，检索top50

python test_dpr-Linq-Embed-Mistral.py