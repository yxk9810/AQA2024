import json
import re

data_test = [json.loads(line.strip()) for line in open("qa_test_wo_ans_new.txt","r")]
data = data_test
def fielter_url(text):
    text = text.replace("\n","")
    text = re.sub(r'<.*?>',"",text)
    text = " ".join([x for x in text.strip().split(" ") if not ("<" in x or "/" in x or ">" in x)])
    return text

with open("question2body.json","w",encoding="utf-8") as f:
    for x in data:
        q = x["question"]
        body = fielter_url(x["body"])
        js = json.dumps({"question":q,"body":body},ensure_ascii=False)
        f.write("{}\n".format(js))