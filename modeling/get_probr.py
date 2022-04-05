import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModel
from dataset import Dataset

def main():
    test_dataset = Dataset("../data/socialIQA/socialIQa_v1.4_tst.jsonl", 'test', "bert-base-uncased", False)
    test_dataloader = DataLoader(test_dataset, batch_size=1)
    model_path = sys.argv[1]
    model_r = AutoModel.from_pretrained(model_path).cuda()
    classifier_r = torch.load(f"{model_path}/classifier.pt").cuda()
    model_r.eval()
    m = nn.Softmax(dim=0)
    for step, eval_batch in enumerate(test_dataloader):
        eval_batch_r = eval_batch['cqs']
        for key in eval_batch_r:
            eval_batch_r[key] = eval_batch_r[key][0].cuda()
        with torch.no_grad():
            outputs_r = model_r(input_ids=eval_batch_r["input_ids"], attention_mask=eval_batch_r["attention_mask"])
        pooled_output_r = outputs_r[1]
        logits_r = classifier_r(pooled_output_r)
        probs_r = m(logits_r)
        probs_r = torch.squeeze(probs_r)
        top5 = torch.topk(probs_r, 5).indices
        #print(step, top5.tolist())
        print(step, probs_r.tolist())

main()

