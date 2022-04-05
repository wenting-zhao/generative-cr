import json
import sys
from transformers import T5Tokenizer, T5ForConditionalGeneration

model_name = "allenai/unifiedqa-t5-large" # you can specify the model size here
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

def run_model(input_string, **generator_args):
    input_ids = tokenizer.encode(input_string, return_tensors="pt")
    res = model.generate(input_ids, **generator_args)
    return tokenizer.batch_decode(res, skip_special_tokens=True)


filename = sys.argv[1]
data = []
with open(filename, 'r') as fin:
    for line in fin:
        data.append(json.loads(line))

for d in data:
    cont = d['context']
    q = d['question']
    a = d['answerA']
    b = d['answerB']
    c = d['answerC']
    print(cont)
    print(q)
    print(a)
    print(b)
    print(c)
    cmd = f"{q} \\n (a) {a} (b) {b} (c) {c} \\n {cont}"
    print(cmd)
    while cmd != "":
        print(run_model(cmd))
        cmd = input("new answers: ")
