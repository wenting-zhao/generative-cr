import sys
from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration
import torch

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large", use_fast=True)
model = BartForConditionalGeneration.from_pretrained(sys.argv[1])

while True:
    in_text = input("input to the model:")
    in_tokenized = tokenizer(in_text, return_tensors="pt")["input_ids"]
    out_text = input("output from the model:")
    out_tokenized = tokenizer(out_text, return_tensors="pt")["input_ids"]
    loss = model(input_ids=in_tokenized, labels=out_tokenized).loss
    print("loss is", loss.mean())
