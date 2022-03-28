import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch
import nltk
from nltk.tag.stanford import StanfordNERTagger
import sys


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 128
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            all_scores = []
            for batch in tqdm(list(chunks(examples, self.batch_size))):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    return_dict_in_generate=True,
                    output_scores=True
                    )
                scores = []
                for i in range(len(summaries.scores)):
                    scores.append(torch.max(torch.nn.functional.softmax(summaries.scores[i], dim=-1), dim=-1).values.view(1,-1).T)
                scores = torch.cat(scores, 1)
                #print(scores.shape, "111")
                #print(scores, "111")
                #print(summaries.sequences.shape, "2222")
                #print(summaries.sequences, "2222")
                #print(summaries.sequences[..., 1:])
                scores = torch.where(summaries.sequences[..., 1:] <= 1, torch.Tensor([1.]).cuda(), scores)
                all_scores.append(scores.cpu().tolist())
                #print(scores, "333")
                #print(all_scores[-1])
                dec = self.tokenizer.batch_decode(summaries.sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs, all_scores


all_relations = [
    #"AtLocation",
    #"CapableOf",
    #"Causes",
    #"CausesDesire",
    #"CreatedBy",
    #"DefinedAs",
    #"DesireOf",
    #"Desires",
    #"HasA",
    #"HasFirstSubevent",
    #"HasLastSubevent",
    #"HasPainCharacter",
    #"HasPainIntensity",
    #"HasPrerequisite",
    #"HasProperty",
    #"HasSubEvent",
    #"HasSubevent",
    #"HinderedBy",
    #"InheritsFrom",
    #"InstanceOf",
    #"IsA",
    #"LocatedNear",
    #"LocationOfAction",
    #"MadeOf",
    #"MadeUpOf",
    #"MotivatedByGoal",
    #"NotCapableOf",
    #"NotDesires",
    #"NotHasA",
    #"NotHasProperty",
    #"NotIsA",
    #"NotMadeOf",
    #"ObjectUse",
    #"PartOf",
    #"ReceivesAction",
    #"RelatedTo",
    #"SymbolOf",
    #"UsedFor",
    #"isAfter",
    #"isBefore",
    #"isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
    ]

#def get_names(st, text):
#    for sent in nltk.sent_tokenize(text):
#        tokens = nltk.tokenize.word_tokenize(sent)
#        tags = st.tag(tokens)
#        res = []
#        for tag in tags:
#            if tag[1]=='PERSON':
#                res.append(tag[0])
#    return res
def get_names(string):
    for x in string.split()[1:]:
        if x[0] == x[0].upper():
            return x

if __name__ == "__main__":

    # sample usage
    print("model loading ...")
    comet = Comet("./comet-atomic_2020_BART")
    comet.model.zero_grad()
    print("model loaded")

    queries = []
    #head = "PersonX eats an apple"
    #rel = "xNeed"
    #query = "{} {} [GEN]".format(head, rel)
    #queries.append(query)
    #print(queries)
    #results = comet.generate(queries, decode_method="beam", num_generate=5)
    #print(results)
    #st = StanfordNERTagger('stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz', 'stanford-ner/stanford-ner.jar')

    data = []
    #with open("../data/socialIQA/socialIQa_v1.4_tst.jsonl", 'r') as fin:
    with open("../data/cosmosQA/valid.jsonl", 'r') as fin:
        for line in fin:
            data.append(json.loads(line))

    #idx = int(sys.argv[1])
    #for d in [data[idx]]:
    for d in data:
        #head = f"{d['context']} </s> {d['question']} </s> {get_names(d['question'])}"
        head = f"{d['context']}"
        #name = get_names(d['question'])
        #if name is not None: head = head.replace(name, 'PersonX')
        for rel in all_relations:
            query = "{} {} [GEN]".format(head, rel)
            queries.append(query)

    #results = comet.generate(queries, decode_method="beam", num_generate=5)
    results = comet.generate(queries, decode_method="beam", num_generate=1)
    torch.save(results, "res.pt")
    #print(data[idx])
    #for i, res in enumerate(results):
    #    print(all_relations[i], res)

