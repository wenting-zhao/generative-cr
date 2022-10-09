import sys
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", use_fast=True)
model = BartForConditionalGeneration.from_pretrained(sys.argv[1])
data = []
with open("../data/alphanli/test.jsonl", 'r') as fin:
    for line in fin:
        data.append(json.loads(line))

for d in data:
    print(f"obs1: {d['obs1']}")
    print(f"obs2: {d['obs2']}")
    in_tokenized = tokenizer(d['obs1']+d['hyp1'], return_tensors="pt")["input_ids"]
    out_tokenized = tokenizer(d['obs2'], return_tensors="pt")["input_ids"]
    loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
    print("loss for", "\""+d['hyp1']+"\"", "is", loss.mean().item())
    in_tokenized = tokenizer(d['obs1']+d['hyp2'], return_tensors="pt")["input_ids"]
    loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
    print("loss for", "\""+d['hyp2']+"\"", "is", loss.mean().item())
    while True:
        hyp = input("e: ")
        if hyp == "":
            break
        combined = d['obs1'] + hyp
        in_tokenized = tokenizer(combined, return_tensors="pt")["input_ids"]
        loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
        print("loss is", loss.mean().item())
