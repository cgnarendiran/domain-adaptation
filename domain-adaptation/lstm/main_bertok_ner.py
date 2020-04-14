# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import data
import model_ner

import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import glob
import numpy as np
from tqdm import tqdm
from utils_ner import convert_examples_to_features, get_labels, read_examples_from_file

from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)

# Inside my model training code
import wandb
wandb.init(project="domain-adaptation")

logger = logging.getLogger(__name__)


parser = argparse.ArgumentParser(description='PyTorch Enron RNN/LSTM Language Model')
parser.add_argument('--data_dir', type=str, default='./data/conll/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=700,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--learning_rate', type=float, default=0.5,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=512,
                    help='sequence length')
parser.add_argument("--eval_batch_size", default=8, type=int, 
                    help="Batch size per GPU/CPU for evaluation.")
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
# parser.add_argument('--logging_steps', type=int, default=200, metavar='N',
#                     help='report interval')
parser.add_argument('--finetuned_lm_model', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument("--load_pretrained", action="store_true", help="Use a pretrained lm model")
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')
parser.add_argument(
        "--output_dir",
        default='output/',
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.")
parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument("--local_rank", type=int, default=-1, 
                    help="For distributed training: local_rank")
parser.add_argument("--overwrite_cache", action="store_true", 
                    help="Overwrite the cached training and evaluation sets")
parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument("--do_predict", action="store_true", help="Whether to run predictions on the test set.")
parser.add_argument(
    "--evaluate_during_training",
    action="store_true",
    help="Whether to run evaluation during training at each logging step.",
)
parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.")
parser.add_argument(
    "--eval_all_checkpoints",
    action="store_true",
    help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
)
args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

args.device = torch.device("cuda" if args.cuda else "cpu")
args.n_gpu = torch.cuda.device_count()


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


###############################################################################
# Load data
###############################################################################

# corpus = data.Corpus(args.data)

# # Starting from sequential data, batchify arranges the dataset into columns.
# # For instance, with the alphabet as the sequence and batch size 4, we'd get
# # ┌ a g m s ┐
# # │ b h n t │
# # │ c i o u │
# # │ d j p v │
# # │ e k q w │
# # └ f l r x ┘.
# # These columns are treated as independent by the model, which means that the
# # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# # batch processing.


# def batchify(data, bsz):
#     # Work out how cleanly we can divide the dataset into bsz parts.
#     nbatch = data.size(0) // bsz
#     # Trim off any extra elements that wouldn't cleanly fit (remainders).
#     data = data.narrow(0, 0, nbatch * bsz)
#     # Evenly divide the data across the bsz batches.
#     data = data.view(bsz, -1).t().contiguous()
#     return data.to(device)

# eval_batch_size = 16
# train_data = batchify(corpus.train, args.batch_size)
# val_data = batchify(corpus.valid, eval_batch_size)
# print("\nNumber of examples to train on:", train_data.size(0)*args.batch_size//args.bptt)
# print("\nNumber of validation examples:", val_data.size(0)*eval_batch_size//args.bptt)
# # test_data = batchify(corpus.test, eval_batch_size)



def load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}".format(
            mode, list(filter(None, args.model.split("/"))).pop(), str(args.bptt)
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        examples = read_examples_from_file(args.data_dir, mode)
        features = convert_examples_to_features(
            examples,
            labels,
            args.bptt,
            tokenizer,
            # cls_token_at_end=bool(args.model_type in ["xlnet"]),
            # xlnet has a cls token at the end
            cls_token_at_end=False,
            cls_token=tokenizer.cls_token,
            # cls_token_segment_id=2 if args.model_type in ["xlnet"] else 0,
            cls_token_segment_id=0,
            sep_token=tokenizer.sep_token,
            # sep_token_extra=bool(args.model_type in ["roberta"]),
            # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
            sep_token_extra=False,
            # pad_on_left=bool(args.model_type in ["xlnet"]),
            # pad on the left for xlnet
            pad_on_left=False,
            pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
            # pad_token_segment_id=4 if args.model_type in ["xlnet"] else 0,
            pad_token_segment_id=0,
            pad_token_label_id=pad_token_label_id,
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset






def train(args, train_dataset, model, tokenizer, labels, pad_token_label_id):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    # args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model, "optimizer.pt")) and os.path.isfile(
        os.path.join(args.model, "scheduler.pt")
    ):
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(torch.load(os.path.join(args.model, "optimizer.pt")))
        scheduler.load_state_dict(torch.load(os.path.join(args.model, "scheduler.pt")))


    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
        )

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model):
        # set global_step to gobal_step of last saved checkpoint from model path
        try:
            global_step = int(args.model.split("-")[-1].split("/")[0])
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except:
            pass
        

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproductibility
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        hidden = model.init_hidden(args.batch_size)
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            hidden = repackage_hidden(hidden)
            inputs = {"input_ids": batch[0], "attention_mask": batch[2], "hidden": hidden, "attention_mask": batch[1], "labels": batch[3]}
            # if args.model != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "xlnet"] else None
            #     )  # XLM and RoBERTa don"t use segment_ids
            inputs["token_type_ids"] = None

            outputs, hidden = model(**inputs)
            loss = outputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # if args.fp16:
            #     with amp.scale_loss(loss, optimizer) as scaled_loss:
            #         scaled_loss.backward()
            # else:
            #     loss.backward()
            loss.backward()

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # if args.fp16:
                #     torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                # else:
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                scheduler.step()  # Update learning rate schedule
                optimizer.step()
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev")
                        for key, value in results.items():
                            tb_writer.add_scalar("eval_{}".format(key), value, global_step)
                            wandb.log({key: value})
                    tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
                    wandb.log({"learning rate": scheduler.get_lr()[0]})
                    tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
                    wandb.log({"Train loss": (tr_loss - logging_loss) / args.logging_steps})
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), output_dir)
                    # model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    logger.info("Saving model checkpoint to %s", output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    logger.info("Saving optimizer and scheduler states to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, labels, pad_token_label_id, mode, prefix=""):
    eval_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode=mode)

    # args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation %s *****", prefix)
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = None
    out_label_ids = None
    model.eval()
    hidden = model.init_hidden(args.eval_batch_size)
    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            hidden = repackage_hidden(hidden)
            inputs = {"input_ids": batch[0], "hidden":hidden, "attention_mask": batch[1], "labels": batch[3]}
            # if args.model_type != "distilbert":
            #     inputs["token_type_ids"] = (
            #         batch[2] if args.model_type in ["bert", "xlnet"] else None
            #     )  # XLM and RoBERTa don"t use segment_ids
            inputs["token_type_ids"] = batch[2]
            outputs, hidden = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]

            if args.n_gpu > 1:
                tmp_eval_loss = tmp_eval_loss.mean()  # mean() to average on multi-gpu parallel evaluating

            eval_loss += tmp_eval_loss.item()
        nb_eval_steps += 1
        if preds is None:
            preds = logits.detach().cpu().numpy()
            out_label_ids = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    preds = np.argmax(preds, axis=2)

    label_map = {i: label for i, label in enumerate(labels)}

    out_label_list = [[] for _ in range(out_label_ids.shape[0])]
    preds_list = [[] for _ in range(out_label_ids.shape[0])]

    for i in range(out_label_ids.shape[0]):
        for j in range(out_label_ids.shape[1]):
            if out_label_ids[i, j] != pad_token_label_id:
                out_label_list[i].append(label_map[out_label_ids[i][j]])
                preds_list[i].append(label_map[preds[i][j]])

    results = {
        "loss": eval_loss,
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list),
    }

    logger.info("***** Eval results %s *****", prefix)
    for key in sorted(results.keys()):
        logger.info("  %s = %s", key, str(results[key]))

    return results, preds_list













# Prepare CONLL-2003 task
labels = get_labels('')
nlabels = len(labels)
# Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

# Load pretrained model and tokenizer
if args.local_rank not in [-1, 0]:
    torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab



# Build Bert Tokenizer:
tokenizer_class = BertTokenizer
tokenizer = tokenizer_class.from_pretrained("bert-base-cased", cache_dir=args.data_dir)
dictionary = tokenizer.vocab

train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")



###############################################################################
# Build the model
###############################################################################

ntokens = len(tokenizer.vocab)
if args.model == 'Transformer':
    model = model_ner.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(args.device)
else:
    model = model_ner.RNNModel(args.model, ntokens, nlabels, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(args.device)


###############################################################################
# state dict:
###############################################################################
if args.load_pretrained:
    pretrained_dict = torch.load(args.finetuned_lm_model).state_dict()
    model_dict = model.state_dict()

    # print(model_dict.keys())
    # print(pretrained_dict.keys())
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 

    # 3. load the new state dict
    model.load_state_dict(model_dict)


criterion = nn.CrossEntropyLoss()

###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""

    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.

# def get_batch(source, i):
#     seq_len = min(args.bptt, len(source) - 1 - i)
#     data = source[i:i+seq_len]
#     target = source[i+1:i+1+seq_len].view(-1)
#     return data, target


# def evaluate(data_source):
#     # Turn on evaluation mode which disables dropout.
#     model.eval()
#     total_loss = 0.
#     ntokens = len(corpus.dictionary)
#     if args.model != 'Transformer':
#         hidden = model.init_hidden(eval_batch_size)
#     with torch.no_grad():
#         for i in range(0, data_source.size(0) - 1, args.bptt):
#             data, targets = get_batch(data_source, i)
#             if args.model == 'Transformer':
#                 output = model(data)
#             else:
#                 output, hidden = model(data, hidden)
#                 hidden = repackage_hidden(hidden)
#             output_flat = output.view(-1, ntokens)
#             total_loss += len(data) * criterion(output_flat, targets).item()
#     return total_loss / (len(data_source) - 1)


# def train():
#     # Turn on training mode which enables dropout.
#     model.train()
#     total_loss = 0.
#     start_time = time.time()
#     ntokens = len(tokenizer.vocab)
#     # ntokens = len(tokenizer.vocab)
#     if args.model != 'Transformer':
#         hidden = model.init_hidden(args.batch_size)
#     for batch, i in tqdm(enumerate(range(0, train_data.size(0) - 1, args.bptt)), desc="Steps"):
#         data, targets = get_batch(train_data, i)
#         # Starting each batch, we detach the hidden state from how it was previously produced.
#         # If we didn't, the model would try backpropagating all the way to start of the dataset.
#         model.zero_grad()
#         if args.model == 'Transformer':
#             output = model(data)
#         else:
#             hidden = repackage_hidden(hidden)
#             output, hidden = model(data, hidden)
#         loss = criterion(output.view(-1, ntokens), targets)
#         loss.backward()

#         # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
#         torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
#         for p in model.parameters():
#             p.data.add_(-lr, p.grad.data)

#         total_loss += loss.item()

#         if batch % args.log_interval == 0 and batch > 0:
#             cur_loss = total_loss / args.log_interval
#             elapsed = time.time() - start_time
#             print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
#                     'loss {:5.2f} | ppl {:8.2f}'.format(
#                 epoch, batch, len(train_data) // args.bptt, lr,
#                 elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
#             # wandb logging:
#             wandb.log({"learning_rate": lr})
#             wandb.log({'training_loss': cur_loss, "training_perplexity": math.exp(cur_loss)}, step=epoch)
#             total_loss = 0
#             start_time = time.time()


# def export_onnx(path, batch_size, seq_len):
#     print('The model is also exported in ONNX format at {}'.
#           format(os.path.realpath(args.onnx_export)))
#     model.eval()
#     dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
#     hidden = model.init_hidden(batch_size)
#     torch.onnx.export(model, (dummy_input, hidden), path)


# Loop over epochs.
# lr = args.lr
# best_val_loss = None

# At any point you can hit Ctrl + C to break out of training early.
# try:
#     for epoch in tqdm(range(1, args.epochs+1), desc="Epochs"):
#         epoch_start_time = time.time()
#         train()
#         val_loss = evaluate(val_data)
#         print('-' * 89)
#         print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
#                 'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
#                                            val_loss, math.exp(val_loss)))
#         print('-' * 89)
#         wandb.log({'validation_loss': val_loss, "validation_perplexity": math.exp(val_loss)}, step=epoch)
#         # Save the model if the validation loss is the best we've seen so far.
#         if not best_val_loss or val_loss < best_val_loss:
#             with open(args.save, 'wb') as f:
#                 torch.save(model, f)
#             best_val_loss = val_loss
#         else:
#             # Anneal the learning rate if no improvement has been seen in the validation dataset.
#             lr /= 4.0
# except KeyboardInterrupt:
#     print('-' * 89)
#     print('Exiting from training early')

# # Load the best saved model.
# with open(args.save, 'rb') as f:
#     model = torch.load(f)
#     # after load the rnn params are not a continuous chunk of memory
#     # this makes them a continuous chunk, and will speed up forward pass
#     # Currently, only rnn model supports flatten_parameters function.
#     if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
#         model.rnn.flatten_parameters()

# Run on test data.
# test_loss = evaluate(test_data)
# print('=' * 89)
# print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
#     test_loss, math.exp(test_loss)))
# print('=' * 89)

# if len(args.onnx_export) > 0:
#     # Export the model in ONNX format.
#     export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)

# model.to(args.device)

logger.info("Training/evaluation parameters %s", args)

# Training
if args.do_train:
    train_dataset = load_and_cache_examples(args, tokenizer, labels, pad_token_label_id, mode="train")
    global_step, tr_loss = train(args, train_dataset, model, tokenizer, labels, pad_token_label_id)
    logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

# Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
    # Create output directory if needed
    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    logger.info("Saving model checkpoint to %s", args.output_dir)
    # Save a trained model, configuration and tokenizer using `save_pretrained()`.
    # They can then be reloaded using `from_pretrained()`
    model_to_save = (
        model.module if hasattr(model, "module") else model
    )  # Take care of distributed/parallel training
    model_to_save.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Good practice: save your training arguments together with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

# Evaluation
results = {}
if args.do_eval and args.local_rank in [-1, 0]:
    # tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    checkpoints = [args.output_dir]
    if args.eval_all_checkpoints:
        checkpoints = list(
            os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True))
        )
        logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
    logger.info("Evaluate the following checkpoints: %s", checkpoints)
    for checkpoint in checkpoints:
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        # model = model_class.from_pretrained(checkpoint)
        model.load_state_dict(torch.load(checkpoint).state_dict())
        model.to(args.device)
        result, _ = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="dev", prefix=global_step)
        if global_step:
            result = {"{}_{}".format(global_step, k): v for k, v in result.items()}
        results.update(result)
    output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        for key in sorted(results.keys()):
            writer.write("{} = {}\n".format(key, str(results[key])))

if args.do_predict and args.local_rank in [-1, 0]:
    tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case)
    # model = model_class.from_pretrained(args.output_dir)
    model.load_state_dict(torch.load(args.output_dir).state_dict())
    model.to(args.device)
    result, predictions = evaluate(args, model, tokenizer, labels, pad_token_label_id, mode="test")
    # Save results
    output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
    with open(output_test_results_file, "w") as writer:
        for key in sorted(result.keys()):
            writer.write("{} = {}\n".format(key, str(result[key])))
    # Save predictions
    output_test_predictions_file = os.path.join(args.output_dir, "test_predictions.txt")
    with open(output_test_predictions_file, "w") as writer:
        with open(os.path.join(args.data_dir, "test.txt"), "r") as f:
            example_id = 0
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    writer.write(line)
                    if not predictions[example_id]:
                        example_id += 1
                elif predictions[example_id]:
                    output_line = line.split()[0] + " " + predictions[example_id].pop(0) + "\n"
                    writer.write(output_line)
                else:
                    logger.warning("Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0])
