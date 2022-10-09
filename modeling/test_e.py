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
    return round(tensor.item(), 5)

def compute_acc(preds, ls):
    preds = torch.argmin(preds, -1)
    preds = preds.view(-1).tolist()
    correct = []
    assert len(preds) == len(ls)
    for p, l in zip(preds, ls):
        if p == l:
            correct.append(1)
        else:
            correct.append(0)
    return sum(correct)/len(ls), correct

def pretty_print(i, first, second):
    print(f"obs1: {data[i]['obs1']}")
    print(f"obs2: {data[i]['obs2']}")
    print(f"answer: {labels[i]}")
    print(f"loss for \"{data[i][hyp1]}\" is {fm(first[i][0])}, {fm(second[i][0])}")
    print(f"loss for \"{data[i][hyp2]}\" is {fm(first[i][1])}, {fm(second[i][1])}")
    print()

out1 = torch.load("logging/used_for_paper/model-bart-large lr-1e-06 b-8 reg-0 option-0|step-0.pt")
out2 = torch.load("logging/used_for_paper/model-bart-large lr-1e-06 b-8 reg-0 option-0|step-1000.pt")
out3 = torch.load('logging/model-bart-large lr-1e-06 b-8 reg-10.0 option-0|step-89828.pt')
acc1, raw1 = compute_acc(out1, labels)
acc2, raw2 = compute_acc(out2, labels)
acc3, raw3 = compute_acc(out3, labels)
print("ZS LVM: ", acc1)
print("LL LVM: ", acc2)
print("PR LVM: ", acc3)
corrected_by_ll = []
corrected_by_pr = []
ll_correct_not_pr = []
zs_correct_not_ll = []
for i in range(len(raw1)):
    if raw1[i] == 0 and raw2[i] == 1:
        corrected_by_ll.append(i)
    elif raw1[i] == 1 and raw2[i] == 0:
        zs_correct_not_ll.append(i)
    if raw2[i] == 0 and raw3[i] == 1:
        corrected_by_pr.append(i)
    elif raw2[i] == 1 and raw3[i] == 0:
        ll_correct_not_pr.append(i)

hyp1 = "hyp1"
hyp2 = "hyp2"
print(f"here are {len(corrected_by_ll)} instances corrected by ll")
for i in corrected_by_ll:
    pretty_print(i, out1, out2)
print(f"here are {len(zs_correct_not_ll)} instances are corrected with zs but not ll")
for i in zs_correct_not_ll:
    pretty_print(i, out1, out2)
print(f"here are {len(corrected_by_pr)} instances corrected by pr")
for i in corrected_by_pr:
    pretty_print(i, out2, out3)
print(f"here are {len(ll_correct_not_pr)} instances are corrected with ll but not pr")
for i in ll_correct_not_pr:
    pretty_print(i, out2, out3)
#for d, l, p1, p2, p3 in zip(data, labels, out1, out2, out3):
#    print(f"obs1: {d['obs1']}")
#    print(f"obs2: {d['obs2']}")
#    print(f"answer: {l}")
#    hyp1 = "hyp1"
#    hyp2 = "hyp2"
#    #in_tokenized = tokenizer(d['obs1']+d['hyp1'], return_tensors="pt")["input_ids"]
#    #out_tokenized = tokenizer(d['obs2'], return_tensors="pt")["input_ids"]
#    #loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
#    #loss2 = model2(input_ids=in_tokenized, labels=out_tokenized).loss
#    #loss3 = model3(input_ids=in_tokenized, labels=out_tokenized).loss
#    print(f"loss for \"{d[hyp1]}\" is {fm(p1[0])}, {fm(p2[0])}, {fm(p3[0])}")
#    #in_tokenized = tokenizer(d['obs1']+d['hyp2'], return_tensors="pt")["input_ids"]
#    #loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
#    #loss2 = model2(input_ids=in_tokenized, labels=out_tokenized).loss
#    #loss3 = model3(input_ids=in_tokenized, labels=out_tokenized).loss
#    print(f"loss for \"{d[hyp2]}\" is {fm(p1[1])}, {fm(p2[1])}, {fm(p3[1])}")
#    print()
#    #while True:
#    #    hyp = input("e: ")
#    #    if hyp == "":
#    #        break
#    #    combined = d['obs1'] + hyp
#    #    in_tokenized = tokenizer(combined, return_tensors="pt")["input_ids"]
#    #    loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
#    #    print("loss is", loss.mean().item())
