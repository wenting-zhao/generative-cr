import sys
import torch

data = torch.load(sys.argv[1])
pred_file = sys.argv[2]
preds = []
with open(pred_file, 'r') as f:
    for line in f:
        sidx = line.find('[')
        eidx = line.find(']')
        pred = torch.Tensor([float(x) for x in line[sidx+1:eidx].split(', ')])
        preds.append(pred)
answers = torch.load("cache_siqa/answers.pt")
a_probs = torch.load("cache_siqa/res_proba.pt")

def main():
    for i, item in enumerate(data):
        a_prob = a_probs[i].view(3, -1)
        if 'cst_label1' in item and 'cst_label2' in item:
            if len(item['cst_label2']) != 0:
                continue
        print(i)
        print(item['context'])
        print(item['question'])
        print(item['answerA'])
        print(item['answerB'])
        print(item['answerC'])
        print(item['correct'], answers[i])
        if item['correct'] == 'A':
            a = 0
        elif item['correct'] == 'B':
            a = 1
        elif item['correct'] == 'C':
            a = 2
        else:
            raise ValueError(f"{item['correct']} is not valid")
        a_prob = a_prob[a]
        pred = preds[i]
        a_sum = a_prob.sum()
        ps = []
        for j in range(len(a_prob)):
            #p = pred[j] * (a_prob[j] / a_sum)
            p = a_prob[j] / a_sum
            ps.append(p.item())
        ps = torch.Tensor(ps)
        top3 = torch.topk(ps, 3).indices.tolist()
        print("posterior:", torch.topk(ps, 3).values.tolist())
        prior_top3 = torch.topk(pred, 3).indices.tolist()
        print("prior:", torch.topk(pred, 3).values.tolist())
        score = [round(sum(z)/len(z),2) for z in item['score']]
        print(score)
        print([round(z,2) for z in pred.tolist()])
        for j, cst in enumerate(item['cst']):
            if j in top3:
                print(j, cst, "-"*30)
            #if j in top3 and j not in prior_top3:
            #    print(j, cst, "-"*30)
            #elif j in top3 and j in prior_top3:
            #    print(j, cst, "-"*30, "#"*30)
            #elif j not in top3 and j in prior_top3:
            #    print(j, cst, "#"*30)
            else:
                print(j, cst)
        item['cst_pred'] = top3
        tmp = input("relevant relations (seperated by space): ")
        item['cst_label1'] = [int(y) for y in tmp.split()]
        tmp2 = input("relevant knowledge (seperated by space): ")
        item['cst_label2'] = [int(y) for y in tmp2.split()]
        print()
    torch.save(data, "cache/test_data_complete.pt")

try:
    main()
except KeyboardInterrupt:
    for i, item in enumerate(data):
        if 'cst_label1' not in item and 'cst_label2' not in item:
            torch.save(data, f"cache/test_data_{i}.pt")
            break
