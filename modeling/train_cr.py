import argparse
from copy import deepcopy
import logging
from itertools import chain
from dataclasses import dataclass
from typing import Optional, Union
import math
from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoModel
from transformers import AdamW, SchedulerType
from transformers import get_scheduler, set_seed
from transformers.file_utils import PaddingStrategy
from datasets import load_metric
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from dataset import Dataset
from tqdm.auto import tqdm
from torch.nn import NLLLoss
from torch.distributions import Categorical
import wandb
wandb.login()

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        #cqs = [feature.pop("cqs") for feature in features]
        #scores = []
        #for feature in features:
        #    scores += feature.pop("scores")
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        #batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        #batch["cqs"] = cqs
        #batch["scores"] = torch.tensor([np.prod(l) for l in scores], dtype=torch.float)
        #batch["scores"] = torch.tensor([1/np.sum(l)/len(l) for l in scores], dtype=torch.float)
        #batch["scores"] = batch["scores"].view(-1, 1)
        return batch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--test_path", type=str)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument("--minimum", default=0.62, type=float,
                        help="minimum acc to start run test.")
    parser.add_argument("--batch_size", '-b', default=1, type=int,
                        help="batch size per gpu.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="eval batch size per gpu.")
    parser.add_argument("--epoch", '-epoch', default=10, type=int,
                        help="The number of epochs for fine-tuning.")
    parser.add_argument("--num_choice", default=3, type=int,
                        help="The number of choices in QA.")
    parser.add_argument("--num_cst", default=51, type=int,
                        help="The number of commonsense triplets.")
    parser.add_argument("--model_dir", default="roberta-large", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--output_model_dir", default="./saved_models", type=str,
                        help="The directory where the pretrained model will be saved.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Warmup ratio in the lr scheduler."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="linear",
        help="The scheduler type to use.",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    args = parser.parse_args()
    return args

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(555)
    logger = logging.getLogger(__name__)

    args = get_args()
    train_dataset = Dataset(args.train_path, 'train', args.model_dir, args.baseline)
    eval_dataset = Dataset(args.valid_path, 'dev', args.model_dir, args.baseline)
    test_dataset = Dataset(args.test_path, 'test', args.model_dir, args.baseline)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    data_collator = DataCollatorForMultipleChoice(tokenizer, padding='longest')
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    def get_model_optim():
        model = AutoModel.from_pretrained(args.model_dir)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
        return model, optimizer

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps

    model_a, optim_a = get_model_optim()
    dropout_a = nn.Dropout(model_a.config.hidden_dropout_prob)
    classifier_a = nn.Linear(model_a.config.hidden_size, 1)
    classifier_a = classifier_a.to(device)
    model_a.to(device)
    lr_scheduler_a = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optim_a,
        num_warmup_steps=int(args.warmup_ratio*args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )

    #if not args.baseline:
    #    model_r, optim_r = get_model_optim()
    #    dropout_r = nn.Dropout(model_r.config.hidden_dropout_prob)
    #    classifier_r = nn.Linear(model_r.config.hidden_size, 1)
    #    classifier_r = classifier_r.to(device)
    #    model_r.to(device)
    #    lr_scheduler_r = get_scheduler(
    #        name=args.lr_scheduler_type,
    #        optimizer=optim_r,
    #        num_warmup_steps=args.num_warmup_steps,
    #        num_training_steps=args.max_train_steps,
    #    )
    #model_a = AutoModel.from_pretrained("saved_models_a").cuda()
    #model_r = AutoModel.from_pretrained("saved_models_r").cuda()
    #classifier_a = torch.load(f"saved_models_a/classifier.pt").cuda()
    #classifier_r = torch.load(f"saved_models_r/classifier.pt").cuda()

    loss_fct = NLLLoss()
    m = nn.Softmax(dim=0)
    m1 = nn.LogSoftmax(dim=-1)

    def compute_logits(outputs, b=None, train=True, outputs_r=None):
        pooled_output = outputs[1]
        if train:
            pooled_output = dropout_a(pooled_output)
        logits = classifier_a(pooled_output)
        if not args.baseline:
            #pooled_output_r = outputs_r[1]
            #if train:
            #    pooled_output_r = dropout_r(pooled_output_r)
            #logits_r = classifier_r(pooled_output_r)
            #probs_r = m(logits_r)
            probs = m(logits)
            probs = probs.T #* b['scores'].view(1, -1)
            #reshaped_probs = probs.view(-1, args.num_choice, args.num_cst)
            reshaped_probs = probs.view(1, args.num_choice, -1)
            #print(reshaped_probs)
            #reshaped_probs = torch.permute(reshaped_probs, (0, 2, 1))
            #reshaped_probs = reshaped_probs * probs_r
            #reshaped_probs = torch.permute(reshaped_probs, (0, 2, 1))
            reshaped_probs = torch.sum(reshaped_probs, dim=-1)
            #return reshaped_probs, probs_r
            return torch.log(reshaped_probs)
        else:
            return m1(logits.view(-1, args.num_choice))
 
    metric = load_metric("accuracy")

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    name = "baseline" if args.baseline else "generative"
    if not args.nolog:
        wandb.init(name=f'{name} training (joint): model-{args.model_dir} lr-{args.learning_rate} b-{args.batch_size*args.gradient_accumulation_steps} l2-{args.weight_decay}',
               project='generative cr',
               #project='generative cr cosmosQA',
               tags=['SocialIQA'])
               #tags=['cosmosQA'])
        wandb.config.lr = args.learning_rate
        wandb.watch(model_a)
        #if not args.baseline:
        #    wandb.watch(model_r)
    best_valid = 0

    def evaluate(dataloader, split):
        model_a.eval()
        #res = []
        #if not args.baseline:
        #    model_r.eval()
        for step, eval_batch in enumerate(dataloader):
            #eval_batch_r = deepcopy(eval_batch['cqs'][0])
            #del eval_batch['cqs']
            #for key in eval_batch_r:
            #    eval_batch_r[key] = eval_batch_r[key].to(device)
            for key in eval_batch:
                eval_batch[key] = eval_batch[key].to(device)
            with torch.no_grad():
                outputs = model_a(input_ids=eval_batch["input_ids"], attention_mask=eval_batch["attention_mask"])
                #if not args.baseline:
                #    outputs_r = model_r(input_ids=eval_batch_r["input_ids"], attention_mask=eval_batch_r["attention_mask"])
            if not args.baseline:
                #probs, r_probs = compute_logits(outputs, eval_batch, False, outputs_r)
                probs = compute_logits(outputs, eval_batch, False)
            else:
                probs = compute_logits(outputs, train=False)
            predictions = probs.argmax(dim=-1)
            #pooled_output = outputs[1]
            #logits = classifier_a(pooled_output)
            #probs2 = m(logits)
            #res.append(predictions)
            #res.append(probs2)
            metric.add_batch(
                predictions=predictions,
                references=eval_batch["labels"],
            )

        eval_metric = metric.compute()
        #torch.save(res, "res.pt")
        if not args.nolog:
            wandb.log({
                "step": completed_steps,
                f"{split} Acc": eval_metric})
        return eval_metric['accuracy']

    for epoch in range(args.epoch):
        model_a.train()
        #if not args.baseline:
        #    model_r.train()
        for step, batch in enumerate(train_dataloader):
            #evaluate(test_dataloader, "Test")
            #return
            #batch_r = deepcopy(batch['cqs'][0])
            #del batch['cqs']
            #for key in batch_r:
            #    batch_r[key] = batch_r[key].to(device)
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model_a(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            if not args.baseline:
                #outputs_r = model_r(input_ids=batch_r["input_ids"], attention_mask=batch_r["attention_mask"])
                #probs, r_probs = compute_logits(outputs, batch, outputs_r=outputs_r)
                probs = compute_logits(outputs, batch)
            else:
                probs = compute_logits(outputs)
            loss = loss_fct(probs, batch['labels'])
            #nll = loss_fct(torch.log(probs), batch['labels'])
            #ent_loss = Categorical(raw).entropy().mean()
            #loss = nll + ent_loss
            #loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optim_a.step()
                lr_scheduler_a.step()
                optim_a.zero_grad()
                #if not args.baseline:
                #    optim_r.step()
                #    lr_scheduler_r.step()
                #    optim_r.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if not args.nolog:
                    wandb.log({
                        "step": completed_steps,
                        "Train Loss": loss.item()})

            if step % (500*args.gradient_accumulation_steps) == 0:
                acc = evaluate(eval_dataloader, "Valid")
                if acc > best_valid and acc > args.minimum:
                    evaluate(test_dataloader, "Test")
                    best_valid = acc
                    model_a.save_pretrained(f"{args.output_model_dir}_a")
                    torch.save(classifier_a, f"{args.output_model_dir}_a/classifier.pt")
                    if not args.baseline:
                        model_r.save_pretrained(f"{args.output_model_dir}_r")
                        torch.save(classifier_r, f"{args.output_model_dir}_r/classifier.pt")

if __name__ == '__main__':
    main()
