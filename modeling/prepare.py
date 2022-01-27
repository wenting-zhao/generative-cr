import json
import torch
from transformers import RobertaTokenizer
from datasets import load_dataset
from generation_example import all_relations


def preprocess_function(examples, tokenizer, ending_names):
    num_answers = len(ending_names)
    num_triplets = len(examples[0]['cst'])
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    first_sentences = [[f"{e['context']} {t}" for t in e['cst']] * num_answers for e in examples]
    first_sentences = sum(first_sentences, [])

    # Grab all second sentences possible for each context.
    second_sentences = [[f"{examples[i][end]}"] * num_triplets for i in range(len(examples)) for end in ending_names]
    second_sentences = sum(second_sentences, [])
    assert len(first_sentences) == len(second_sentences)

    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True, padding=True)
    # Un-flatten
    return {k: [v[i:i+num_answers*num_triplets] for i in range(0, len(v), num_answers*num_triplets)] for k, v in tokenized_examples.items()}

def main():
    ending_names = ["answerA", "answerB", "answerC"]
    data = []
    with open("../data/socialIQA/socialIQa_v1.4_trn.jsonl", 'r') as fin:
        for line in fin:
            data.append(json.loads(line))

    raw0, raw1 = torch.load("res.pt")
    raw0 = [item for sub in raw0 for item in sub]
    raw1 = [item for sub in raw1 for item in sub]
    num_examples = len(data) 
    csts = []
    scores = []
    split = len(raw0)//num_examples
    for i in range(0, len(raw0), split):
        csts.append([x for j, x in enumerate(raw0[i:i+split])])
        scores.append([x for j, x in enumerate(raw1[i:i+split])])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', use_fast=True)
    assert len(data) == len(csts) == len(scores)
    for example, cst in zip(data, csts):
        example['cst'] = [x+y for x, y in zip(all_relations, cst)]
    features = preprocess_function(data[:2], tokenizer, ending_names)
    features['scores'] = [s * len(ending_names) for s in scores[:2]]
    torch.save(features, "aaa.pt")
    #features['scores'] = scores[:2]
    #for item, cst in zip(data, csts):
    #    label = item['correct']
    #    if label == 'A':
    #        item['label'] = 0
    #    elif label == 'B':
    #        item['label'] = 1
    #    elif label == 'C':
    #        item['label'] = 2

if __name__ == '__main__':
    main()
