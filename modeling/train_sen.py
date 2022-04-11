import argparse
import wandb
from itertools import chain
from dataset import SenMakingDataset
from datasets import load_metric
from dataclasses import dataclass
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Optional, Union
from transformers.file_utils import PaddingStrategy
from transformers import get_scheduler, set_seed
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers import BartForConditionalGeneration, GPTNeoForCausalLM
from transformers import AdamW, SchedulerType
import numpy as np
import math
from tqdm import tqdm
import sys
wandb.login()

@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        targets = [[feature.pop("targets")] * num_choices for feature in features]
        targets = list(chain(*targets))
        targets = [{"input_ids": x} for x in targets]
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
        batch_target = self.tokenizer.pad(
            targets,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )['input_ids']
        batch_target[batch_target==self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)] = -100

        # Un-flatten
        #batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        batch["targets"] = batch_target
        return batch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--test_path", type=str)
    parser.add_argument('--baseline', action='store_true')
    parser.add_argument('--supervised', action='store_true')
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--nolog', action='store_true')
    parser.add_argument('--save_model', action='store_true')
    parser.add_argument("--batch_size", '-b', default=1, type=int,
                        help="batch size per gpu.")
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="eval batch size per gpu.")
    parser.add_argument("--epoch", '-epoch', default=10, type=int,
                        help="The number of epochs for fine-tuning.")
    parser.add_argument("--num_choice", default=2, type=int,
                        help="The number of choices in QA.")
    parser.add_argument("--num_cst", default=51, type=int,
                        help="The number of commonsense triplets.")
    parser.add_argument("--model_dir", default="roberta-large", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--last_checkpoint", default="", type=str,
                        help="The directory where the pretrained model will be loaded.")
    parser.add_argument("--output_model_dir", default="./saved_models_sen", type=str,
                        help="The directory where the pretrained model will be saved.")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay to use.")
    parser.add_argument(
        "--warmup_ratio", type=float, default=0, help="Warmup ratio in the lr scheduler."
    )
    parser.add_argument(
        "--reg_coeff", type=float, default=0, help="Coefficient for regularizer."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
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
    if args.last_checkpoint == "":
        args.last_checkpoint = args.model_dir
    return args

def main():
    metric = load_metric("accuracy")
    def evaluate(dataloader, split):
        model.eval()
        ents = []
        if args.sample:
            out = []
        for step, eval_batch in enumerate(dataloader):
            bs = len(eval_batch['targets'])
            for key in eval_batch:
                eval_batch[key] = eval_batch[key].to(device)
            if args.baseline:
                concated = torch.cat((eval_batch["input_ids"], eval_batch["targets"]), dim=1)
                concated_label = torch.cat((eval_batch["input_ids"], eval_batch["targets"]), dim=1)
                concated_label[concated_label==model.config.pad_token_id] = -100
                with torch.no_grad():
                    outputs = model(input_ids=concated, labels=concated_label).loss
            else:
                with torch.no_grad():
                    outputs = model(input_ids=eval_batch["input_ids"], attention_mask=eval_batch["attention_mask"], labels=eval_batch["targets"]).loss
            outputs = outputs.view(bs, -1).mean(dim=-1)
            outputs = outputs.view(-1, 3)
            normalized = m(outputs)
            entropy = torch.mean(-torch.sum(normalized * torch.log(normalized + 1e-9), dim = 1), dim = 0)
            ents.append(entropy.cpu().item())
            predictions = outputs.argmin(dim=-1)
            metric.add_batch(
                predictions=predictions,
                references=eval_batch["labels"],
            )
            if args.sample:
                out.append(outputs.cpu())
        if args.sample:
            torch.save(torch.cat(out, dim=0), f"logging/{run_name}|step-{completed_steps}.pt")
        eval_metric = metric.compute()
        if not args.baseline:
            if not args.nolog:
                wandb.log({
                    "step": completed_steps,
                    f"{split} Loss": outputs.mean(),
                    f"{split} Reg": np.mean(ents),
                    f"{split} Acc": eval_metric})
        return eval_metric['accuracy']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(555)

    args = get_args()
    train_dataset = SenMakingDataset(args.train_path, args.model_dir)
    eval_dataset = SenMakingDataset(args.valid_path, args.model_dir)
    test_dataset = SenMakingDataset(args.test_path, args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if args.baseline:
        model = GPTNeoForCausalLM.from_pretrained(args.model_dir)
    else:
        model = BartForConditionalGeneration.from_pretrained(args.last_checkpoint)
    model = model.to(device)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    m = nn.Softmax(dim=-1)

    data_collator = DataCollatorForMultipleChoice(tokenizer, padding='longest')
    train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.batch_size)
    eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)
    test_dataloader = DataLoader(test_dataset, collate_fn=data_collator, batch_size=args.eval_batch_size)

    model_name = args.model_dir.split('/')[-1]
    if args.supervised:
        run_name=f'supervised-model-{model_name} lr-{args.learning_rate} b-{args.batch_size*args.gradient_accumulation_steps} reg-{args.reg_coeff}'
    else:
        run_name=f'model-{model_name} lr-{args.learning_rate} b-{args.batch_size*args.gradient_accumulation_steps} reg-{args.reg_coeff}'

    if args.baseline:
        print("valid:", evaluate(eval_dataloader, "Valid"))
        print("test:", evaluate(test_dataloader, "Test"))
        sys.exit(0)

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
    optim = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    loss_fct = nn.CrossEntropyLoss()

    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.epoch * num_update_steps_per_epoch
    step_size = args.reg_coeff / args.max_train_steps
    total_batch_size = args.batch_size * args.gradient_accumulation_steps
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optim,
        num_warmup_steps=int(args.warmup_ratio*args.max_train_steps),
        num_training_steps=args.max_train_steps,
    )

    progress_bar = tqdm(range(args.max_train_steps))
    completed_steps = 0

    if not args.nolog:
        wandb.init(name=run_name,
               project='generative sense making',
               tags=['sense'])
        wandb.config.lr = args.learning_rate
        wandb.watch(model)

    best_valid = float('-inf')
    for epoch in range(args.epoch):
        model.train()
        for step, batch in enumerate(train_dataloader):
            if step % (500*args.gradient_accumulation_steps) == 0:
                valid_acc = evaluate(eval_dataloader, "Valid")
                evaluate(test_dataloader, "Test")
                if valid_acc > best_valid:
                    best_valid = valid_acc
                    if args.save_model:
                        model.save_pretrained(f"{args.output_model_dir}/{run_name}")
            bs = len(batch['targets'])
            for key in batch:
                batch[key] = batch[key].to(device)
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch['targets']).loss
            reshaped_outputs = outputs.view(bs, -1).mean(dim=-1)
            normalized = m(reshaped_outputs.view(-1, 3))
            entropy = args.reg_coeff * torch.mean(-torch.sum(normalized * torch.log(normalized + 1e-9), dim = 1), dim = 0)
            args.reg_coeff -= step_size
            if args.supervised:
                loss = -loss_fct(reshaped_outputs.view(-1, 3), batch['labels'])
            else:
                loss = outputs.mean()
            tot_loss = loss + entropy
            tot_loss.backward()
            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                optim.step()
                lr_scheduler.step()
                optim.zero_grad()
                progress_bar.update(1)
                completed_steps += 1
                if not args.nolog:
                    wandb.log({
                        "step": completed_steps,
                        "Reg": entropy.item(),
                        "Train Loss": loss.item()})


if __name__ == '__main__':
    main()
