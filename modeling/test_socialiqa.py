import sys
from transformers import AutoTokenizer
from transformers import AutoModel
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("roberta-large", use_fast=True)
model = AutoModel.from_pretrained(sys.argv[1]).cuda()
classifier = torch.load(f"{sys.argv[1]}/classifier.pt").cuda()
data = []
with open("../data/socialIQA/socialIQa_v1.4_tst.jsonl", 'r') as fin:
    for line in fin:
        data.append(json.loads(line))

for d in data:
    cont = d['context']
    q = d['question']
    a = d['answerA']
    b = d['answerB']
    c = d['answerC']
    in_text = [f"{cont} {tokenizer.sep_token} {q} {tokenizer.sep_token} {x}" for x in [a, b, c]]
    in_tokenized = tokenizer(in_text, return_tensors="pt", padding="longest")
    with torch.no_grad():
        output = model(input_ids=in_tokenized['input_ids'].cuda(), attention_mask=in_tokenized['attention_mask'].cuda())
    pooled_output = output[1]
    logits = classifier(pooled_output).view(1, -1).cpu().tolist()[0]
    print(cont)
    print(q)
    print(a, logits[0])
    print(b, logits[1])
    print(c, logits[2])
    while True:
        a = input("a: ")
        if a == "":
            break
        combined = f"{cont} {tokenizer.sep_token} {q} {tokenizer.sep_token} {a}"
        in_tokenized = tokenizer(combined, return_tensors="pt")["input_ids"].cuda()
        with torch.no_grad():
            output = model(input_ids=in_tokenized)
        pooled_output = output[1]
        logits = classifier(pooled_output)
        print(logits.item())
