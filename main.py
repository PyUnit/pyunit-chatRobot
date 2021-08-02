#!/usr/bin/python3.8
# -*- coding: utf-8 -*-
# @Time  : 2019/12/27 17:48
# @Author: Jtyoui@qq.com
import os
import torch
import random
import uvicorn
from itertools import chain
from argparse import ArgumentParser
import torch.nn.functional as F
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from transformers import OpenAIGPTLMHeadModel, GPT2LMHeadModel, BertTokenizer

app = FastAPI(title='闲聊机器人', description='闲聊机器人接口文档', version='1.0')
MODEL = os.path.join(os.path.dirname(__file__), 'model')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

parser = ArgumentParser()
parser.add_argument('--gpt2', action='store_true', help="use gpt2")
parser.add_argument("--model_checkpoint", type=str, default=MODEL, help="Path, url or short name of the model")
parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
parser.add_argument("--max_length", type=int, default=30, help="Maximum length of the output utterances")
parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
parser.add_argument("--seed", type=int, default=42, help="Seed")
parser.add_argument("--temperature", type=int, default=0.7, help="Sampling softmax temperature")
parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
parser.add_argument("--top_p", type=float, default=0.9)
args = parser.parse_args()

SPECIAL_TOKENS = ["[CLS]", "[SEP]", "[PAD]", "[speaker1]", "[speaker2]"]


def top_filtering(logics, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logics.dim() == 1
    top_k = min(top_k, logics.size(-1))
    if top_k > 0:
        indices_to_remove = logics < torch.topk(logics, top_k)[0][..., -1, None]
        logics[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logics, sorted_indices = torch.sort(logics, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logics, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probabilities > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logics[indices_to_remove] = filter_value
    indices_to_remove = logics < threshold
    logics[indices_to_remove] = filter_value
    return logics


def build_input_from_segments(history, reply, tokenizer, with_eos=True):
    bos, eos, pad, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    sequence = [[bos]] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if i % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    instance = {"input_ids": list(chain(*sequence)),
                "token_type_ids": [bos] + [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence[1:])
                                           for _ in s]}
    return instance, sequence


def sample_sequence(history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(history, current_output, tokenizer, with_eos=False)
        input_ids = torch.tensor(instance["input_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], dtype=torch.long, device=args.device).unsqueeze(0)
        logics, *_ = model(input_ids, token_type_ids=token_type_ids)
        logics = logics[0, -1, :] / args.temperature
        logics = top_filtering(logics, top_k=args.top_k, top_p=args.top_p)
        props = F.softmax(logics, dim=-1)
        prev = torch.topk(props, 1)[1] if args.no_sample else torch.multinomial(props, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            while prev.item() in special_tokens_ids:
                prev = torch.multinomial(props, num_samples=1)
        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())
    return current_output


random.seed(42)
torch.random.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
tokenizer_class = BertTokenizer
model_class = OpenAIGPTLMHeadModel if not args.gpt2 else GPT2LMHeadModel
tokenizer = tokenizer_class.from_pretrained(MODEL, do_lower_case=True)
model = model_class.from_pretrained(MODEL)
model.to(args.device)
model.eval()
history = []


def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)


@app.get('/robot/chat', summary='实体接口')
async def chat(data: str = Query(..., description='文本数据', min_length=1)):
    try:
        global history
        raw_text = " ".join(list(data.replace(" ", "")))
        history.append(tokenize(raw_text))
        with torch.no_grad():
            out_ids = sample_sequence(history, tokenizer, model, args)
        history.append(out_ids)
        history = history[-(2 * args.max_history + 1):]
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True).replace(" ", "")
        if len(history) > 50:
            history = []
        return {'code': 200, 'data': out_text}
    except Exception as e:
        return {'code': 400, 'msg': str(e)}


if __name__ == '__main__':
    uvicorn.run(app=app, host="0.0.0.0", port=80)
