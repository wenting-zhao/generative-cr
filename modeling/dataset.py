import json
import pickle
import os
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer
from datasets import load_dataset
from generation_example import all_relations


def preprocess_function(examples, tokenizer, ending_names, baseline=False):
    num_answers = len(ending_names)
    num_triplets = len(examples[0]['cst'])
    # Repeat each first sentence four times to go with the four possibilities of second sentences.
    if baseline:
        first_sentences = [[f"{e['context']}"] * num_answers for e in examples]
    else:
        first_sentences = [[f"{e['context']} {t}" for t in e['cst']] * num_answers for e in examples]
    first_sentences = sum(first_sentences, [])

    # Grab all second sentences possible for each context.
    if baseline:
        second_sentences = [[f"{e[end]}"] for e in examples for end in ending_names]
    else:
        second_sentences = [[f"{e[end]}"] * num_triplets for e in examples for end in ending_names]
    second_sentences = sum(second_sentences, [])
    assert len(first_sentences) == len(second_sentences)

    print("start to tokenize")
    # Tokenize
    tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
    # Un-flatten
    print("done tokenize")
    if baseline:
        return {k: [v[i:i+num_answers] for i in range(0, len(v), num_answers)] for k, v in tokenized_examples.items()}
    else:
        return {k: [v[i:i+num_answers*num_triplets] for i in range(0, len(v), num_answers*num_triplets)] for k, v in tokenized_examples.items()}

def prepare(filename, split, baseline=False):
    print("preparing ", filename)
    ending_names = ["answerA", "answerB", "answerC"]
    data = []
    with open(filename, 'r') as fin:
        for line in fin:
            data.append(json.loads(line))

    raw0, raw1 = torch.load(f"cache/res_{split}.pt")
    raw0 = [item for sub in raw0 for item in sub]
    raw1 = [item for sub in raw1 for item in sub]
    num_examples = len(data) 
    csts = []
    scores = []
    interval = len(raw0)//num_examples
    for i in range(0, len(raw0), interval):
        csts.append([x for j, x in enumerate(raw0[i:i+interval])])
        scores.append([x for j, x in enumerate(raw1[i:i+interval])])

    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', use_fast=True)
    assert len(data) == len(csts) == len(scores)
    for example, cst in zip(data, csts):
        example['cst'] = [x+y for x, y in zip(all_relations, cst)]
    if os.path.isfile(f"cache/{split}_encodings.pkl"):
        with open(f"cache/{split}_encodings.pkl", 'rb') as f:
            features = pickle.load(f)
    else:
        features = preprocess_function(data[:1000], tokenizer, ending_names, baseline)
        with open(f"cache/{split}_encodings.pkl", 'wb') as f:
            pickle.dump(features, f)
    features['scores'] = [s * len(ending_names) for s in scores[:1000]]
    labels = []
    for item in data[:1000]:
        label = item['correct']
        if label == 'A':
            labels.append(0)
        elif label == 'B':
            labels.append(1)
        elif label == 'C':
            labels.append(2)
    return features, labels

class Dataset(torch.utils.data.Dataset):
    def __init__(self, filename, split, baseline=False):
        self.encodings, self.labels = prepare(filename, split, baseline)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

