import sys
import re
import numpy as np
import torch

data = torch.load(sys.argv[1])
answers = torch.load("cache/answers.pt")

correct_r = 0
correct_o = 0
b_correct_r = 0
b_correct_o = 0
b_correct_r2 = 0
b_correct_o2 = 0
correct_o = 0
tot = 0

pred_correct = 0
pred_correct_cst_correct = 0
for i, item in enumerate(data):
    if 'cst_label2' in item:
        if answers[i] == 0 and item['correct'] == 'A' or answers[i] == 1 and item['correct'] == 'B' or answers[i] == 2 and item['correct'] == 'C':
            pred_correct += 1
        tot += 1
        for j in item['cst_pred']:
            if j in item['cst_label1']:
                correct_r += 1
                if len(set(item['cst_pred']).intersection(set(item['cst_label2']))) > 0:
                    correct_o += 1
                if answers[i] == 0 and item['correct'] == 'A' or answers[i] == 1 and item['correct'] == 'B' or answers[i] == 2 and item['correct'] == 'C':
                    pred_correct_cst_correct += 1
                break
        bpred = torch.topk(torch.tensor([np.prod(l) for l in item['score']]), 3).indices.tolist()
        if len(set(bpred).intersection(set(item['cst_label2']))) > 0:
            b_correct_o2 += 1
            b_correct_r2 += 1
        if len(re.findall(r'What will .* want to do next', item['question'])) > 0:
            for j in item['cst_label1']:
                if "xWant" in item['cst'][j] or "oWant" in item['cst'][j] or "hasSubEvent" in item['cst'][j]:
                    b_correct_r += 1
                    if len(set(item['cst_label1']).intersection(set(item['cst_label2']))) > 0:
                        b_correct_o += 1
                    break
        elif len(re.findall(r'What would .* feel afterwards', item['question'])) > 0:
            for j in item['cst_label1']:
                if "xReact" in item['cst'][j] or "oReact" in item['cst'][j] or "Cause" in item['cst'][j]:
                    b_correct_r += 1
                    if len(set(item['cst_label1']).intersection(set(item['cst_label2']))) > 0:
                        b_correct_o += 1
                    break
        elif len(re.findall(r'What would you describe X', item['question'])) > 0:
            for j in item['cst_label1']:
                if "xAttr" in item['cst'][j]:
                    b_correct_r += 1
                    if len(set(item['cst_label1']).intersection(set(item['cst_label2']))) > 0:
                        b_correct_o += 1
                    break
        elif len(re.findall(r'Why did .* do this', item['question'])) > 0:
            for j in item['cst_label1']:
                if "xReason" in item['cst'][j] or "xIntent" in item['cst'][j] or "HinderedBy" in item['cst'][j]:
                    b_correct_r += 1
                    if len(set(item['cst_label1']).intersection(set(item['cst_label2']))) > 0:
                        b_correct_o += 1
                    break
        elif len(re.findall(r'What does .* need to do before this', item['question'])) > 0:
            for j in item['cst_label1']:
                if "xNeed" in item['cst'][j] or "isAfter" in item['cst'][j] or "isFilledBy" in item['cst'][j]:
                    b_correct_r += 1
                    if len(set(item['cst_label1']).intersection(set(item['cst_label2']))) > 0:
                        b_correct_o += 1
                    break
        elif len(re.findall(r'What will happen to .*', item['question'])) > 0:
            for j in item['cst_label1']:
                if "xEffect" in item['cst'][j] or "oEffect" in item['cst'][j] or "isBefore" in item['cst'][j]:
                    b_correct_r += 1
                    if len(set(item['cst_label1']).intersection(set(item['cst_label2']))) > 0:
                        b_correct_o += 1
                    break

print("correct r:", correct_r / tot)
print("baseline correct r:", b_correct_r / tot)
print("baseline correct r2:", b_correct_r2 / tot)
print("correct o:", correct_o / tot)
print("baseline correct o:", b_correct_o / tot)
print("baseline correct o2:", b_correct_o2 / tot)
print("correct / correct:", pred_correct_cst_correct / pred_correct, pred_correct_cst_correct, pred_correct)
