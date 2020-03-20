# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import data
import model

import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import glob
import numpy as np

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
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

parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/enron/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
parser.add_argument('--emsize', type=int, default=700,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=1024,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=0.5,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=512,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')

parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--onnx-export', type=str, default='',
                    help='path to export the final model in onnx format')

parser.add_argument('--nhead', type=int, default=2,
                    help='the number of heads in the encoder/decoder of the transformer model')
parser.add_argument(
        "--train_data_file", default="./data/enron/train.txt", type=str, required=False, help="The input training data file (a text file)."
    )
parser.add_argument(
        "--model_type", type=str, default="bert", required=False, help="The model architecture to be trained or fine-tuned.",
    )
parser.add_argument(
        "--output_dir",
        type=str,
        default= "./output/",
        required=False,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
parser.add_argument(
        "--eval_data_file",
        default="./data/enron/valid.txt",
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
parser.add_argument(
        "--model_name_or_path",
        default="bert-base-cased",
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

parser.add_argument(
        "--cache_dir",
        default="./data/enron/",
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
parser.add_argument(
        "--block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
parser.add_argument(
        "--evaluate_during_training", action="store_true", help="Run evaluation during training at each logging step."
    )

parser.add_argument("--per_gpu_train_batch_size", default=4, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument(
        "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
    )
parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
parser.add_argument(
        "--num_train_epochs", default=20, type=float, help="Total number of training epochs to perform."
    )
parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

parser.add_argument("--logging_steps", type=int, default=500, help="Log every X updates steps.")
parser.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every X updates steps.")
parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name_or_path ending and ending with step number",
    )
parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
parser.add_argument(
        "--overwrite_output_dir", action="store_true", help="Overwrite the content of the output directory"
    )
parser.add_argument(
        "--overwrite_cache", action="store_true", help="Overwrite the cached training and evaluation sets"
    )
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument("--server_ip", type=str, default="", help="For distant debugging.")
parser.add_argument("--server_port", type=str, default="", help="For distant debugging.")
args = parser.parse_args()



class TextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        assert os.path.isfile(file_path)

        block_size = block_size - (tokenizer.max_len - tokenizer.max_len_single_sentence)

        directory, filename = os.path.split(file_path)
        cached_features_file = os.path.join(
            directory, args.model_type + "_cached_lm_" + str(block_size) + "_" + filename
        )

        if os.path.exists(cached_features_file) and not args.overwrite_cache:
            logger.info("Loading features from cached file %s", cached_features_file)
            with open(cached_features_file, "rb") as handle:
                self.examples = pickle.load(handle)
        else:
            logger.info("Creating features from dataset file at %s", directory)

            self.examples = []
            with open(file_path, encoding="utf-8") as f:
                text = f.read()

            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))

            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
            # Note that we are loosing the last truncated example here for the sake of simplicity (no padding)
            # If your dataset is small, first you should loook for a bigger one :-) and second you
            # can change this behavior by adding (model specific) padding.

            logger.info("Saving features into cached file %s", cached_features_file)
            with open(cached_features_file, "wb") as handle:
                pickle.dump(self.examples, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item], dtype=torch.long)



def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)



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

def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

def get_batch_modified(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args):
    labels = inputs.clone()
    return inputs, labels

def evaluate(data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    if args.model != 'Transformer':
        hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
            if args.model == 'Transformer':
                output = model(data)
            else:
                output, hidden = model(data, hidden)
                hidden = repackage_hidden(hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


def train(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    start_time = time.time()
    # ntokens = len(corpus.dictionary)
    ntokens = len(tokenizer.vocab)
    
    # for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
    #     data, targets = get_batch(train_data, i)
    #     # Starting each batch, we detach the hidden state from how it was previously produced.
    #     # If we didn't, the model would try backpropagating all the way to start of the dataset.
    #     model.zero_grad()
    #     if args.model == 'Transformer':
    #         output = model(data)
    #     else:
    #         hidden = repackage_hidden(hidden)
    #         output, hidden = model(data, hidden)
    #     loss = criterion(output.view(-1, ntokens), targets)
    #     loss.backward()

    #     # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
    #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
    #     for p in model.parameters():
    #         p.data.add_(-lr, p.grad.data)

    #     total_loss += loss.item()

    #     if batch % args.log_interval == 0 and batch > 0:
    #         cur_loss = total_loss / args.log_interval
    #         elapsed = time.time() - start_time
    #         print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
    #                 'loss {:5.2f} | ppl {:8.2f}'.format(
    #             epoch, batch, len(train_data) // args.bptt, lr,
    #             elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
    #         # wandb logging:
    #         wandb.log({'epoch': epoch, 'training_loss': cur_loss, "training_perplexity": math.exp(cur_loss), "learning_rate": lr})
    #         total_loss = 0
    #         start_time = time.time()

    #
    ###########################################################################################
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate
    )

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

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

    # # Check if saved optimizer or scheduler states exist
    # if (
    #     args.model_name_or_path
    #     and os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt"))
    #     and os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt"))
    # ):
    #     # Load in optimizer and scheduler states
    #     optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "optimizer.pt")))
    #     scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, "scheduler.pt")))

    # if args.fp16:
    #     try:
    #         from apex import amp
    #     except ImportError:
    #         raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
    #     model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    # if args.n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    # if args.local_rank != -1:
    #     model = torch.nn.parallel.DistributedDataParallel(
    #         model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True
    #     )

    # Log metrics with wandb:
    wandb.watch(model)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint
    if args.model_name_or_path and os.path.exists(args.model_name_or_path):
        try:
            # set global_step to gobal_step of last saved checkpoint from model path
            checkpoint_suffix = args.model_name_or_path.split("-")[-1].split("/")[0]
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader) // args.gradient_accumulation_steps)

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info("  Continuing training from epoch %d", epochs_trained)
            logger.info("  Continuing training from global step %d", global_step)
            logger.info("  Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)
        except ValueError:
            logger.info("  Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    # model_to_resize = model.module if hasattr(model, "module") else model  # Take care of distributed/parallel training
    # model_to_resize.resize_token_embeddings(len(tokenizer))

    hidden = model.init_hidden(args.batch_size)
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0]
    )
    set_seed(args)  # Added here for reproducibility


    #     if batch % args.log_interval == 0 and batch > 0:
    #         cur_loss = total_loss / args.log_interval
    #         elapsed = time.time() - start_time
    #         print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
    #                 'loss {:5.2f} | ppl {:8.2f}'.format(
    #             epoch, batch, len(train_data) // args.bptt, lr,
    #             elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
    #         # wandb logging:
    #         wandb.log({'epoch': epoch, 'training_loss': cur_loss, "training_perplexity": math.exp(cur_loss), "learning_rate": lr})
    #         total_loss = 0
    #         start_time = time.time()
    for epoch, _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            data, targets = get_batch(batch, step)
            data = data.to(args.device)
            targets = targets.to(args.device)
            model.train()
            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            output, hidden = model(data, hidden)
            loss = criterion(output.view(-1, ntokens), targets)
            loss.backward()

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            for p in model.parameters():
                p.data.add_(-lr, p.grad.data)

            total_loss += loss.item()

            if step % args.log_interval == 0 and step > 0:
                cur_loss = total_loss / args.log_interval
                elapsed = time.time() - start_time
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                        'loss {:5.2f} | ppl {:8.2f}'.format(
                    epoch, step, len(train_data) // args.bptt, lr,
                    elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
                # wandb logging:
                wandb.log({'epoch': epoch, 'training_loss': cur_loss, "training_perplexity": math.exp(cur_loss), "learning_rate": lr})
                total_loss = 0
                start_time = time.time()
                
            
#####################################################
            # Skip past any already trained steps if resuming training
    #         if steps_trained_in_current_epoch > 0:
    #             steps_trained_in_current_epoch -= 1
    #             continue

    #         # inputs, labels = mask_tokens(batch, tokenizer, args) if args.mlm else (batch, batch)
    #         inputs = inputs.to(args.device)
    #         labels = labels.to(args.device)
    #         model.train()
    #         outputs = model(inputs, masked_lm_labels=labels) if args.mlm else model(inputs, labels=labels)
    #         loss = outputs[0]  # model outputs are always tuple in transformers (see doc)

    #         if args.n_gpu > 1:
    #             loss = loss.mean()  # mean() to average on multi-gpu parallel training
    #         if args.gradient_accumulation_steps > 1:
    #             loss = loss / args.gradient_accumulation_steps

    #         if args.fp16:
    #             with amp.scale_loss(loss, optimizer) as scaled_loss:
    #                 scaled_loss.backward()
    #         else:
    #             loss.backward()

    #         tr_loss += loss.item()
    #         if (step + 1) % args.gradient_accumulation_steps == 0:
    #             if args.fp16:
    #                 torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
    #             else:
    #                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #             optimizer.step()
    #             scheduler.step()  # Update learning rate schedule
    #             model.zero_grad()
    #             global_step += 1

    #             if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
    #                 # Log metrics
    #                 if (
    #                     args.local_rank == -1 and args.evaluate_during_training
    #                 ):  # Only evaluate when single GPU otherwise metrics may not average well
    #                     results = evaluate(args, model, tokenizer)
    #                     for key, value in results.items():
    #                         tb_writer.add_scalar("eval_{}".format(key), value, global_step)
    #                         wandb.log({key: value})
    #                 tb_writer.add_scalar("lr", scheduler.get_lr()[0], global_step)
    #                 wandb.log({"learning rate": scheduler.get_lr()[0]})
    #                 tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, global_step)
    #                 wandb.log({"Train loss": (tr_loss - logging_loss) / args.logging_steps})
    #                 logging_loss = tr_loss

    #             if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
    #                 checkpoint_prefix = "checkpoint"
    #                 # Save model checkpoint
    #                 output_dir = os.path.join(args.output_dir, "{}-{}".format(checkpoint_prefix, global_step))
    #                 os.makedirs(output_dir, exist_ok=True)
    #                 model_to_save = (
    #                     model.module if hasattr(model, "module") else model
    #                 )  # Take care of distributed/parallel training
    #                 model_to_save.save_pretrained(output_dir)
    #                 tokenizer.save_pretrained(output_dir)

    #                 torch.save(args, os.path.join(output_dir, "training_args.bin"))
    #                 logger.info("Saving model checkpoint to %s", output_dir)

    #                 _rotate_checkpoints(args, checkpoint_prefix)

    #                 torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    #                 torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
    #                 logger.info("Saving optimizer and scheduler states to %s", output_dir)

    #         if args.max_steps > 0 and global_step > args.max_steps:
    #             epoch_iterator.close()
    #             break
    #     if args.max_steps > 0 and global_step > args.max_steps:
    #         train_iterator.close()
    #         break

    # if args.local_rank in [-1, 0]:
    #     tb_writer.close()

    # return global_step, tr_loss / global_step






def export_onnx(path, batch_size, seq_len):
    print('The model is also exported in ONNX format at {}'.
          format(os.path.realpath(args.onnx_export)))
    model.eval()
    dummy_input = torch.LongTensor(seq_len * batch_size).zero_().view(-1, batch_size).to(device)
    hidden = model.init_hidden(batch_size)
    torch.onnx.export(model, (dummy_input, hidden), path)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


###############################################################################
# Main code:
###############################################################################

if __name__=="__main__":

    # Set the random seed manually for reproducibility.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    args.device = torch.device("cuda" if args.cuda else "cpu")


    ###############################################################################
    # Load data
    ###############################################################################

    config_class = BertConfig
    model_class = BertForMaskedLM
    tokenizer_class = BertTokenizer
    tokenizer = tokenizer_class.from_pretrained("bert-base-cased", cache_dir=args.cache_dir)
    train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

    # corpus = data.Corpus(args.data)

    # Starting from sequential data, batchify arranges the dataset into columns.
    # For instance, with the alphabet as the sequence and batch size 4, we'd get
    # ┌ a g m s ┐
    # │ b h n t │
    # │ c i o u │
    # │ d j p v │
    # │ e k q w │
    # └ f l r x ┘.
    # These columns are treated as independent by the model, which means that the
    # dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    # batch processing.
    # ntokens = len(corpus.dictionary)


    # eval_batch_size = 16
    # train_data = batchify(corpus.train, args.batch_size)
    # val_data = batchify(corpus.valid, eval_batch_size)
    # test_data = batchify(corpus.test, eval_batch_size)


    ###############################################################################
    # Build the model
    ###############################################################################

    model = model.RNNModel(args.model, args.block_size, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(args.device)

    criterion = nn.CrossEntropyLoss()

    ###############################################################################
    # Training code
    ###############################################################################


    # Loop over epochs.
    lr = args.learning_rate
    best_val_loss = None

    # At any point you can hit Ctrl + C to break out of training early.
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train(args, train_dataset, model, tokenizer)
            ################# take care of this:
            val_loss = evaluate(val_data)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                    'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                               val_loss, math.exp(val_loss)))
            print('-' * 89)
            wandb.log({'validation_loss': val_loss, "validation_perplexity": math.exp(val_loss)})
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(args.save, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
            else:
                # Anneal the learning rate if no improvement has been seen in the validation dataset.
                lr /= 4.0
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()

    # Run on test data.
    test_loss = evaluate(test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)

    if len(args.onnx_export) > 0:
        # Export the model in ONNX format.
        export_onnx(args.onnx_export, batch_size=1, seq_len=args.bptt)
