import json
import pickle
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from generation_example import all_relations

class Dataset(torch.utils.data.Dataset):
    def preprocess_function(self, examples, tokenizer, ending_names, baseline=False):
        num_answers = len(ending_names)
        # Repeat each first sentence four times to go with the four possibilities of second sentences.
        if baseline:
            first_sentences = [[f"{e['context']} {tokenizer.sep_token} {e['question']}"] * num_answers for e in examples]
        else:
            first_sentences = [[f"{e['context']} {tokenizer.sep_token} {t} {tokenizer.sep_token} {e['question']}" for t in e['cst']] * num_answers for e in examples]
        first_sentences = sum(first_sentences, [])

        # Grab all second sentences possible for each context.
        if baseline:
            second_sentences = [[f"{e[end]}"] for e in examples for end in ending_names]
        else:
            second_sentences = [[f"{e[end]}"] * len(e['cst']) for e in examples for end in ending_names]
        lens = [0]
        for i, s in enumerate(second_sentences):
            if i % num_answers == 0:
                last = lens[-1]
                lens.append(len(s)*num_answers+last)

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
            return {k: [v[lens[i]:lens[i+1]] for i in range(len(lens)-1)] for k, v in tokenized_examples.items()}

    def prepare(self, filename, split, path, baseline=False):
        print("preparing ", filename)
        #ending_names = ["answerA", "answerB", "answerC"]
        ending_names = ["answer0", "answer1", "answer2", "answer3"]
        data = []
        with open(filename, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))

        raw0, raw1 = torch.load(f"cache/res_{split}.pt")
        raw0 = [item for sub in raw0 for item in sub]
        raw1 = [item for sub in raw1 for item in sub]
        num_examples = len(data) 
        csts = []
        #scores = []
        interval = len(raw0)//num_examples
        for i in range(0, len(raw0), interval):
            csts.append([x for j, x in enumerate(raw0[i:i+interval])])
            #scores.append([x for j, x in enumerate(raw1[i:i+interval])])

        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        #assert len(data) == len(csts) == len(scores)
        assert len(data) == len(csts)
        #for example, cst, score in zip(data, csts, scores):
        for example, cst in zip(data, csts):
            if_none = [(y.strip() in {'none', ''}) for y in cst]
            example['cst'] = set([f"{self.sym2words[x]}{y}" for x, y, z in zip(all_relations, cst, if_none) if not z])
            if len(example['cst']) == 0: 
                example['cst'] = [tokenizer.sep_token]
            #example['cst'] = [y for y in cst if y.strip() != 'none']
            #example['score'] = [x for x, y, z in zip(score, cst, if_none) if not z]
            #for h in range(len(example['score'])):
            #    example['score'][h] = [hh for hh in example['score'][h] if hh != 1.0]
            #assert len(example['cst']) == len(example['score'])
        if os.path.isfile(f"cache/{split}_encodings.pkl"):
            with open(f"cache/{split}_encodings.pkl", 'rb') as f:
                features = pickle.load(f)
        else:
            features = self.preprocess_function(data, tokenizer, ending_names, baseline)
            with open(f"cache/{split}_encodings.pkl", 'wb') as f:
                pickle.dump(features, f)
        #features['scores'] = [e['score'] * len(ending_names) for e in data]
        #features['cqs'] = [[f"{e['context']} {tokenizer.sep_token} {e['question']} {tokenizer.sep_token} {r}" for c, r in zip(cst, all_relations) if c.strip() != 'none'] for cst, e in zip(csts, data)]
        #features['cqs'] = [tokenizer(l, padding='longest', return_tensors='pt') for l in features['cqs']]
        labels = []
        for item in data:
            # socialIQA
            #label = item['correct']
            #if label == 'A':
            #    labels.append(0)
            #elif label == 'B':
            #    labels.append(1)
            #elif label == 'C':
            #    labels.append(2)
            # COSMOSQA
            labels.append(int(item['label']))
        #torch.save(data, "test_data.pt")
        return features, labels

    def __init__(self, filename, split, path, baseline=False):
        self.sym2words = {"xWant": "PersonX wants", "xReact": "PersonX is", "xNeed": "PersonX needs", "xIntent": "PersonX wants", "xAttr": "PersonX is", "xEffect": "PersonX", "oReact": "PersonX is", "oEffect": "PersonX", "oWant": "PersonX wants", "xReason": "because"}
        self.encodings, self.labels = self.prepare(filename, split, path, baseline)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

class AlphaNLIDataset(torch.utils.data.Dataset):
    def preprocess_function(self, examples, tokenizer, ending_names):
        premise = "obs1"
        first_sentences = [[f"{example[premise]} {example[end]}" for end in ending_names] for example in examples]
        second_sentences = [[example["obs2"]] * len(ending_names) for example in examples]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])

        tokenized_examples = tokenizer(first_sentences, second_sentences, truncation=True)
        return {k: [v[i:i+len(ending_names)] for i in range(0, len(v), len(ending_names))] for k, v in tokenized_examples.items()}

    def prepare(self, filename, path):
        print("preparing ", filename)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        ending_names = ["hyp1", "hyp2"]
        data = []
        labels = []
        with open(filename, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))
        with open(filename.replace(".jsonl", "-labels.lst"), 'r') as fin:
            for idx, line in enumerate(fin):
                labels.append(int(line) - 1)
        if "train" in filename:
            if os.path.isfile(f"cache/anli_encodings.pkl"):
                with open(f"cache/anli_encodings.pkl", 'rb') as f:
                    features = pickle.load(f)
            else:
                features = self.preprocess_function(data, tokenizer, ending_names)
                with open(f"cache/anli_encodings.pkl", 'wb') as f:
                    pickle.dump(features, f)
        else:
            features = self.preprocess_function(data, tokenizer, ending_names)
        return features, labels

    def __init__(self, filename, path):
        self.encodings, self.labels = self.prepare(filename, path)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)
