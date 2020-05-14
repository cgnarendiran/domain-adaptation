# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import numpy as np
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", 10)

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    # BertPreTrainedModel,
    # BertOnlyMLMHead,
    BertModel,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter

# Init wandb
import wandb
wandb.init(project="domain-adaptation")


logger = logging.getLogger(__name__)

class Discriminator(nn.Module):
    """docstring for Discriminator"""
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.config = config
        self.inter_hidden_size = 128
        self.num_domains =  2
        self.main = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.inter_hidden_size),
            nn.Linear(self.inter_hidden_size, self.num_domains-1),
            nn.Sigmoid())

    def forward(self, pooled_hidden_state):
        output = self.main(pooled_hidden_state)
        return output



class BertForAdversarialMaskedLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

        # self.bert = BertModel(config)
        # self.cls = BertOnlyMLMHead(config)
        self.dis = Discriminator(config)
        self.init_weights()

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    # @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        masked_lm_labels=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        lm_labels=None,
        domain_labels=None
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
        lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the left-to-right language modeling loss (next word prediction).
            Indices should be in ``[-100, 0, ..., config.vocab_size]`` (see ``input_ids`` docstring)
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens with labels
            in ``[0, ..., config.vocab_size]``
    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        ltr_lm_loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`lm_labels` is provided):
                Next token prediction loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        Examples::
            from transformers import BertTokenizer, BertForMaskedLM
            import torch
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            model = BertForMaskedLM.from_pretrained('bert-base-uncased')
            input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
            outputs = model(input_ids, masked_lm_labels=input_ids)
            loss, prediction_scores = outputs[:2]
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
        )

        # sequence_output = outputs[0]
        # pooled_output = outputs[1]
        # discriminator_op = self.dis(pooled_output.detach())

        # prediction_scores = self.cls(sequence_output)
        # outputs = (prediction_scores,discriminator_op,) + outputs[2:]  # Add hidden states and attention if they are here

        # adv_loss_fct = nn.BCELoss()

        # # loss for the discriminator:
        # if domain_labels is not None:
        #     # print(discriminator_op)
        #     # print(domain_labels)
        #     dis_loss = adv_loss_fct(discriminator_op.view(-1), domain_labels.view(-1))
        #     outputs = (dis_loss, ) + outputs

        # # loss for the encoder:
        # discriminator_op = self.dis(pooled_output)
        # if domain_labels is not None:
        #     encoder_loss = - adv_loss_fct(discriminator_op.view(-1), domain_labels.view(-1))
        #     outputs = (encoder_loss, ) + outputs

        # # Although this may seem awkward, BertForMaskedLM supports two scenarios:
        # # 1. If a tensor that contains the indices of masked labels is provided,
        # #    the cross-entropy is the MLM cross-entropy that measures the likelihood
        # #    of predictions for masked words.
        # # 2. If `lm_labels` is provided we are in a causal scenario where we
        # #    try to predict the next token for each input in the decoder.
        # if masked_lm_labels is not None:
        #     loss_fct = nn.CrossEntropyLoss()  # -100 index = padding token
        #     masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
        #     outputs = (masked_lm_loss,) + outputs

        # if lm_labels is not None:
        #     # we are doing next-token prediction; shift prediction scores and input ids by one
        #     prediction_scores = prediction_scores[:, :-1, :].contiguous()
        #     lm_labels = lm_labels[:, 1:].contiguous()
        #     loss_fct = nn.CrossEntropyLoss()
        #     ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
        #     outputs = (ltr_lm_loss,) + outputs

        

        return outputs  # (masked_lm_loss), (ltr_lm_loss), (encoder_loss), (dis_loss), prediction_scores, (hidden_states), (attentions)




MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "bert": (BertConfig, BertForAdversarialMaskedLM, BertTokenizer),
    # "bert": (BertConfig, BertForMaskedLM, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForMaskedLM, RobertaTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer),
    "camembert": (CamembertConfig, CamembertForMaskedLM, CamembertTokenizer),
}






class TsvAdvTextDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, args, file_path: str, block_size=512):
        #split the two filepaths
        filepaths = file_path.split(',')
        data = []
        domain = []
        self.examples = []
        for i, filepath in enumerate(filepaths):
            # print(filepath)
            assert os.path.isfile(filepath)
            # Here, we do not cache the features, operating under the assumption
            # that we will soon use fast multithreaded tokenizers from the
            # `tokenizers` repo everywhere =)
            logger.info("Creating features from dataset file at %s", filepath)

            with open(filepath, encoding="utf-8") as f:
                # lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]
                # lines = [line for line in re.split(r'\t+', f.read()) if (len(line) > 0 and not line.isspace())]
                lines_list = pd.read_csv(f, sep='\t').values.tolist()
                lines = [i for l in lines_list for i in l]
            logger.info("Total number of examples in this domain: %s", str(len(lines)))
            # self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
            data.append(tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"])
            # data.append(tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size, pad_to_max_length=True)["input_ids"])
            domain.append([float(i)]*len(data[i]))
            self.examples += list(zip(data[i], domain[i]))


    def __len__(self):
        # return len(self.examples)
        return len(self.examples)

    def __getitem__(self, index):
        # return torch.tensor(self.examples[index])
        return tuple(torch.tensor(example) for example in self.examples[index])




def load_and_cache_examples(args, tokenizer, evaluate=False):
    file_path = args.eval_data_file if evaluate else args.train_data_file
    if args.tsv_adv:
        return TsvAdvTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    elif args.line_by_line:
        return LineByLineTextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)
    else:
        return TextDataset(tokenizer, args, file_path=file_path, block_size=args.block_size)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def _sorted_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> List[str]:
    ordering_and_checkpoint_path = []

    glob_checkpoints = glob.glob(os.path.join(args.output_dir, "{}-*".format(checkpoint_prefix)))

    for path in glob_checkpoints:
        if use_mtime:
            ordering_and_checkpoint_path.append((os.path.getmtime(path), path))
        else:
            regex_match = re.match(".*{}-([0-9]+)".format(checkpoint_prefix), path)
            if regex_match and regex_match.groups():
                ordering_and_checkpoint_path.append((int(regex_match.groups()[0]), path))

    checkpoints_sorted = sorted(ordering_and_checkpoint_path)
    checkpoints_sorted = [checkpoint[1] for checkpoint in checkpoints_sorted]
    return checkpoints_sorted


def _rotate_checkpoints(args, checkpoint_prefix="checkpoint", use_mtime=False) -> None:
    if not args.save_total_limit:
        return
    if args.save_total_limit <= 0:
        return

    # Check if we should delete older checkpoint(s)
    checkpoints_sorted = _sorted_checkpoints(args, checkpoint_prefix, use_mtime)
    if len(checkpoints_sorted) <= args.save_total_limit:
        return

    number_of_checkpoints_to_delete = max(0, len(checkpoints_sorted) - args.save_total_limit)
    checkpoints_to_be_deleted = checkpoints_sorted[:number_of_checkpoints_to_delete]
    for checkpoint in checkpoints_to_be_deleted:
        logger.info("Deleting older checkpoint [{}] due to args.save_total_limit".format(checkpoint))
        shutil.rmtree(checkpoint)


def mask_tokens(inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, args) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, args.mlm_probability)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels





def tsne(args, train_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer):
    tsne = TSNE()
    

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples):
        if tokenizer._pad_token is None:
            return (pad_sequence([d[0] for d in examples], batch_first=True), torch.tensor([d[1] for d in examples]))
        return (pad_sequence([d[0] for d in examples], batch_first=True, padding_value=tokenizer.pad_token_id), torch.tensor([d[1] for d in examples]))

    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, collate_fn=collate)
    # train_dataloader = DataLoader(
    #     train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    model.eval()
    X = None
    y = None
    for batch in tqdm(train_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch[0], tokenizer, args) if args.mlm else (batch[0], batch[0])
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        domain_labels = batch[1].to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels, domain_labels=domain_labels) if args.mlm else model(inputs, labels=labels, domain_labels=domain_labels)
            if X is not None: 
                X = torch.cat((X, outputs[1].item()), 0)
                y = torch.cat((y, domain_labels.item()), 0)
            else: 
                X = outputs[1].item()
                y = domain_labels.item()


    X_embedded = tsne.fit_transform(X)
    sns.scatterplot(X_embedded[:,0], X_embedded[:,1], hue=y, legend='full', palette=palette)



def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, prefix="") -> Dict:
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_output_dir = args.output_dir

    eval_dataset = load_and_cache_examples(args, tokenizer, evaluate=True)

    if args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir, exist_ok=True)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly

    def collate(examples):
        if tokenizer._pad_token is None:
            return (pad_sequence([d[0] for d in examples], batch_first=True), torch.tensor([d[1] for d in examples]))
        return (pad_sequence([d[0] for d in examples], batch_first=True, padding_value=tokenizer.pad_token_id), torch.tensor([d[1] for d in examples]))

    #def collate(examples: List[torch.Tensor]):
    #    if tokenizer._pad_token is None:
    #        return pad_sequence(examples, batch_first=True)
    #    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate
    )

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    eval_dis_loss = 0.0
    nb_eval_steps = 0
    total = 0
    correct = 0
    model.eval()

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = mask_tokens(batch[0], tokenizer, args) if args.mlm else (batch[0], batch[0])
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        domain_labels = batch[1].to(args.device)

        with torch.no_grad():
            outputs = model(inputs, masked_lm_labels=labels, domain_labels=domain_labels) if args.mlm else model(inputs, labels=labels, domain_labels=domain_labels)
            mlm_loss = outputs[0]
            enc_loss = outputs[1]
            dis_loss = outputs[2]
            eval_loss += mlm_loss.mean().item()
            eval_dis_loss += dis_loss.mean().item()

            _, pred_domain_labels = torch.max(outputs[4].data, 1)
            total += domain_labels.size(0)
            correct += (pred_domain_labels == domain_labels).sum().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_dis_loss = eval_dis_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity, "dis eval loss":eval_dis_loss, "dis eval acc":100*correct/total}

    output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")
    with open(output_eval_file, "w") as writer:
        logger.info("***** Eval results {} *****".format(prefix))
        for key in sorted(result.keys()):
            logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))

    return result


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--train_data_file", default=None, type=str, required=True, help="The input training data file (a text file)."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type", type=str, required=True, help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--line_by_line",
        action="store_true",
        help="Whether distinct lines of text in the dataset are to be handled as distinct sequences.",
    )
    parser.add_argument(
        "--tsv_adv",
        action="store_true",
        help="Tab separated lines of text in the dataset. Also specify two different dataset files separated by a comma.",
    )
    parser.add_argument(
        "--should_continue", action="store_true", help="Whether to continue from latest checkpoint in output_dir"
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm", action="store_true", help="Train with masked-language modeling loss instead of language modeling."
    )
    parser.add_argument(
        "--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling loss"
    )

    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--block_size",
        default=-1,
        type=int,
        help="Optional input sequence length after tokenization."
        "The training dataset will be truncated in block of this size for training."
        "Default to the model max input length for single sentence inputs (take into account special tokens).",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_tsne", action="store_true", help="Whether to run training.")
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
        "--num_train_epochs", default=1.0, type=float, help="Total number of training epochs to perform."
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

    if args.model_type in ["bert", "roberta", "distilbert", "camembert"] and not args.mlm:
        raise ValueError(
            "BERT and RoBERTa-like models do not have LM heads but masked LM heads. They must be run using the --mlm "
            "flag (masked language modeling)."
        )
    if args.eval_data_file is None and args.do_eval:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either supply a file to --eval_data_file "
            "or remove the --do_eval argument."
        )
    if args.should_continue:
        sorted_checkpoints = _sorted_checkpoints(args)
        if len(sorted_checkpoints) == 0:
            raise ValueError("Used --should_continue but no checkpoint was found in --output_dir.")
        else:
            args.model_name_or_path = sorted_checkpoints[-1]

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir
            )
        )

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    if args.config_name:
        config = config_class.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        config = config_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        config = config_class()

    if args.tokenizer_name:
        tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name, cache_dir=args.cache_dir)
    elif args.model_name_or_path:
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    else:
        raise ValueError(
            "You are instantiating a new {} tokenizer. This is not supported, but you can do it from another script, save it,"
            "and load it from here, using --tokenizer_name".format(tokenizer_class.__name__)
        )

    if args.block_size <= 0:
        args.block_size = tokenizer.max_len
        # Our input block size will be the max possible for the model
    else:
        args.block_size = min(args.block_size, tokenizer.max_len)

    if args.model_name_or_path:
        model = model_class.from_pretrained(
            args.model_name_or_path,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            cache_dir=args.cache_dir,
        )
    else:
        logger.info("Training new model from scratch")
        model = model_class(config=config)

    print(model.parameters())
    model.to(args.device)

    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # calculating tsne:
    if args.do_tsne:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache

        train_dataset = load_and_cache_examples(args, tokenizer, evaluate=False)

        if args.local_rank == 0:
            torch.distributed.barrier()

        tsne(args, train_dataset, model, tokenizer)


    
if __name__ == "__main__":
    main()
