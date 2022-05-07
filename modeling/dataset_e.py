import json
import pickle
import os
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

class AlphaNLIDataset(torch.utils.data.Dataset):
    def preprocess_function(self, examples, tokenizer, ending_names, option):
        premise = "obs1"
        if option == 0 or option == 4 or option == 5:
            first_sentences = [[f"{example[premise]} {example[end]}" for end in ending_names] for example in examples]
        elif option == 1:
            first_sentences = [[f"{example[premise]}"] for example in examples]
        elif option == 2:
            first_sentences = []
            for example in examples:
                correct = example[ending_names[0]] if example['label'] == 0 else example[ending_names[1]]
                first_sentences.append([f"{example[premise]} {correct}"])
        elif option == 3:
            first_sentences = []
            for example in examples:
                correct = example[ending_names[1]] if example['label'] == 0 else example[ending_names[0]]
                first_sentences.append([f"{example[premise]} {correct}"])
        second_sentences = [[example["obs2"]] for example in examples]
        third_sentences = [[example[premise]] for example in examples]
        fourth_sentences = [[example[end] for end in ending_names] for example in examples]

        first_sentences = sum(first_sentences, [])
        second_sentences = sum(second_sentences, [])
        third_sentences = sum(third_sentences, [])
        fourth_sentences = sum(fourth_sentences, [])

        tokenized_sources = tokenizer(first_sentences, truncation=True)
        tokenized_targets = tokenizer(second_sentences, truncation=True)
        tokenized_sources2 = tokenizer(third_sentences, truncation=True)
        tokenized_targets2 = tokenizer(fourth_sentences, truncation=True)
        length = 1 if 2 <= option <= 3 else len(ending_names)
        tokenized_sources = {k: [v[i:i+length] for i in range(0, len(v), length)] for k, v in tokenized_sources.items()}
        tokenized_targets = {k: [v[i] for i in range(len(v))] for k, v in tokenized_targets.items()}
        tokenized_sources2 = {k: [v[i] for i in range(len(v))] for k, v in tokenized_sources2.items()}
        tokenized_targets2 = {k: [v[i:i+length] for i in range(0, len(v), length)] for k, v in tokenized_targets2.items()}
        return tokenized_sources, tokenized_targets, tokenized_sources2, tokenized_targets2

    def prepare(self, filename, path, option, mask_e):
        print("preparing ", filename)
        tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)
        ending_names = ["hyp1", "hyp2"]
        data = []
        labels = []
        with open(filename, 'r') as fin:
            for line in fin:
                data.append(json.loads(line))
        if option == 4 or option ==5:
            saved_data_path = "cache_e/anli_random_" + filename.split('/')[-1].split('.')[0] + f"_{option}.pt"
            data = torch.load(saved_data_path)
            ending_names = ["hyp1", "hyp2", "hyp3"]
        with open(filename.replace(".jsonl", "-labels.lst"), 'r') as fin:
            for idx, line in enumerate(fin):
                labels.append(int(line) - 1)
                data[idx]['label'] = labels[-1]
        #dedup_data, dedup_labels = [], []
        #obs1ss = []
        #for d, label in zip(data, labels):
        #    if d['obs1'] not in obs1ss:
        #        obs1ss.append(d['obs1'])
        #        dedup_data.append(d)
        #        dedup_labels.append(label)
        #data, labels = dedup_data, dedup_labels

        if "train" in filename:
            if os.path.isfile(f"cache_e/anli_encodings_{option}.pkl"):
                with open(f"cache_e/anli_encodings_{option}.pkl", 'rb') as f:
                    features, targets, features2, targets2 = pickle.load(f)
            else:
                features, targets, features2, targets2 = self.preprocess_function(data, tokenizer, ending_names, option)
                with open(f"cache_e/anli_encodings_{option}.pkl", 'wb') as f:
                    pickle.dump((features, targets, features2, targets2), f)
        else:
            features, targets, features2, targets2 = self.preprocess_function(data, tokenizer, ending_names, option)
        if (option == 2 or option == 3) and mask_e:
            print("masking explanations entirely.")
            ids = tokenizer.convert_tokens_to_ids('.')
            for i in range(len(features['input_ids'])):
                try:
                    #print(i, features['input_ids'][i][0])
                    start = features['input_ids'][i][0].index(ids)
                    end = len(features['input_ids'][i][0]) - 2
                    features['input_ids'][i][0][start+1:end+1] = [tokenizer.mask_token_id] * (end-start)
                    #print(features['input_ids'][i][0])
                except ValueError:
                    # there isn't a period ending obs1
                    print("!!!!!!!!!!!!!!!")
                    pass
        return features, targets, features2, targets2, labels

    def __init__(self, filename, path, option, mask_e=False):
        self.sources, self.targets, self.features2, self.targets2, self.labels = self.prepare(filename, path, option, mask_e)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.sources.items()}
        item['labels'] = self.labels[idx]
        item['targets'] = self.targets['input_ids'][idx]
        item['sources2'] = self.features2['input_ids'][idx]
        item['targets2'] = {key: val[idx] for key, val in self.targets2.items()}
        return item

    def __len__(self):
        return len(self.labels)
