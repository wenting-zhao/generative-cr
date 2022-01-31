import argparse
import logging
from itertools import chain
from dataclasses import dataclass
from typing import Optional, Union
import math
from transformers import RobertaTokenizer, PreTrainedTokenizerBase, RobertaModel
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
from torch.nn import CrossEntropyLoss, BCELoss

@dataclass
class DataCollatorForMultipleChoice:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        scores = []
        for feature in features:
            scores += feature.pop("scores")
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
        batch["scores"] = torch.tensor([np.prod(l) for l in scores], dtype=torch.float)
        batch["scores"] = batch["scores"].view(-1, 1)
        return batch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--test_path", type=str, required=True)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument("--batch_size", '-b', default=16, type=int,
                        help="batch size per gpu.")
    parser.add_argument("--eval_batch_size", default=16, type=int,
                        help="eval batch size per gpu.")
    parser.add_argument("--epoch", '-epoch', default=10, type=int,
                        help="The number of epochs for fine-tuning.")
    parser.add_argument("--num_choice", default=3, type=int,
                        help="The number of choices in QA.")
    parser.add_argument("--num_cst", default=51, type=int,
                        help="The number of commonsense triplets.")
    parser.add_argument("--model_dir", default="roberta-large", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--num_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
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
    set_seed(103)
    logger = logging.getLogger(__name__)

    args = get_args()
    train_dataset = Dataset(args.train_path, 'train', args.baseline)
    eval_dataset = Dataset(args.valid_path, 'dev', args.baseline)
    test_dataset = Dataset(args.test_path, 'test', args.baseline)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_dir)

    data_collator = DataCollatorForMultipleChoice(tokenizer, padding='longest')
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    model = RobertaModel.from_pretrained(args.model_dir)
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
    loss_fct = CrossEntropyLoss()
    dropout = nn.Dropout(model.config.hidden_dropout_prob)
    classifier = nn.Linear(model.config.hidden_size, 1)
    classifier = classifier.to(device)
    sigmoid = nn.Sigmoid()
    model.to(device)

    def compute_logits(outputs):
        pooled_output = outputs[1]
        pooled_output = dropout(pooled_output)
        logits = classifier(pooled_output)
        if not args.baseline:
            probs = sigmoid(logits)
            probs = probs.T * batch['scores'].view(1, -1)
            reshaped_probs = probs.view(-1, args.num_choice, args.num_cst)
            reshaped_probs = torch.sum(reshaped_probs, dim=-1)
            return reshaped_probs
        else:
            return logits.view(-1, args.num_choice)

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    total_batch_size = args.batch_size * args.gradient_accumulation_steps

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.epoch}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total train batch size (w. accumulation) = {total_batch_size}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    for epoch in range(args.epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
            probs = compute_logits(outputs)
            loss = loss_fct(probs, batch['labels'])
            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            if step % (100*args.gradient_accumulation_steps) == 0:
                print(f"global step {completed_steps}: {loss.item()}")
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

            if step % (500*args.gradient_accumulation_steps) == 0:
                model.eval()
                metric = load_metric("accuracy")
                for step, batch in enumerate(eval_dataloader):
                    for key in batch:
                        batch[key] = batch[key].to(device)
                    with torch.no_grad():
                        outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
                    probs = compute_logits(outputs)
                    predictions = probs.argmax(dim=-1)
                    metric.add_batch(
                        predictions=predictions,
                        references=batch["labels"],
                    )

                eval_metric = metric.compute()
                #print(f"epoch {epoch}: {eval_metric}")
                print(f"global step {completed_steps}: {eval_metric}")

if __name__ == '__main__':
    main()
