import sys
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration
import torch
import json

#tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", use_fast=True)
#model = BartForConditionalGeneration.from_pretrained(sys.argv[1])
#model2 = BartForConditionalGeneration.from_pretrained(sys.argv[1])
#model3 = BartForConditionalGeneration.from_pretrained(sys.argv[1])
data = []
with open("../data/alphanli/test.jsonl", 'r') as fin:
    for line in fin:
        data.append(json.loads(line))
labels = []
with open("../data/alphanli/test-labels.lst", 'r') as fin:
    for line in fin:
        labels.append(int(json.loads(line))-1)

def fm(tensor):
    return round(tensor.item(), 2)

def compute_acc(preds, ls):
    preds = torch.argmin(preds, -1)
    preds = preds.view(-1).tolist()
    correct = 0
    for p, l in zip(preds, ls):
        if p == l: correct += 1
    return correct/len(ls)

out1 = torch.load("logging/used_for_paper/model-bart-large lr-1e-06 b-8 reg-0 option-0|step-0.pt")
out2 = torch.load("logging/used_for_paper/model-bart-large lr-1e-06 b-8 reg-0 option-0|step-1000.pt")
out3 = torch.load('logging/model-bart-large lr-1e-06 b-8 reg-10.0 option-0|step-22207.pt')
print(compute_acc(out1, labels))
print(compute_acc(out2, labels))
print(compute_acc(out3, labels))
for d, l, p1, p2, p3 in zip(data, labels, out1, out2, out3):
    print(f"obs1: {d['obs1']}")
    print(f"obs2: {d['obs2']}")
    print(f"answer: {l}")
    #in_tokenized = tokenizer(d['obs1']+d['hyp1'], return_tensors="pt")["input_ids"]
    #out_tokenized = tokenizer(d['obs2'], return_tensors="pt")["input_ids"]
    #loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
    #loss2 = model2(input_ids=in_tokenized, labels=out_tokenized).loss
    #loss3 = model3(input_ids=in_tokenized, labels=out_tokenized).loss
    print("loss for", "\""+d['hyp1']+"\"", "is", fm(p1[0]), fm(p2[0]), fm(p3[0]))
    #in_tokenized = tokenizer(d['obs1']+d['hyp2'], return_tensors="pt")["input_ids"]
    #loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
    #loss2 = model2(input_ids=in_tokenized, labels=out_tokenized).loss
    #loss3 = model3(input_ids=in_tokenized, labels=out_tokenized).loss
    print("loss for", "\""+d['hyp2']+"\"", "is", fm(p1[1]), fm(p2[1]), fm(p3[1]))
    print()
    #while True:
    #    hyp = input("e: ")
    #    if hyp == "":
    #        break
    #    combined = d['obs1'] + hyp
    #    in_tokenized = tokenizer(combined, return_tensors="pt")["input_ids"]
    #    loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
    #    print("loss is", loss.mean().item())
