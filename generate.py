import torch
import json
import os
import random

from tqdm import tqdm
from datasets import load_from_disk, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor
from vllm import LLM, SamplingParams
from argparse import ArgumentParser

from utils import Prompt, PointICL

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

parser = ArgumentParser()
parser.add_argument('--dataset', default='WNUT2017', choices=['CoNLL2003', 'WNUT2017', 'ACE2004', 'ACE2005'], type=str)
parser.add_argument('--model_name', default='Mixtral-8x7B-Instruct-v0.1', type=str)
parser.add_argument('--mode', default='picl+icl', choices=['baseline', 'icl', 'picl+icl'], type=str)
parser.add_argument('--use_bert', action='store_true')
parser.add_argument('--picl_cnt', default=5, type=int)
parser.add_argument('--icl_cnt', default=10, type=int)
parser.add_argument('--batch_cnt', default=16, type=int)
args = parser.parse_args()

model_name = args.model_name
dataset = args.dataset
mode = args.mode
use_bert = args.use_bert
picl_cnt = args.picl_cnt
icl_cnt = args.icl_cnt
batch_cnt = args.batch_cnt

if mode == 'baseline':
    output_path = f"./data/{dataset}/{model_name}-{dataset}-{mode}.json"
elif mode == 'icl':
    output_path = f"./data/{dataset}/{model_name}-{dataset}-{mode}-{icl_cnt}-shot.json"
elif mode == 'picl':
    output_path = f"./data/{dataset}/{model_name}-{dataset}-{mode}-{picl_cnt}-shot.json"
elif mode == 'picl+icl':
    if use_bert:
        output_path = f"./data/{dataset}/{model_name}-{dataset}-{mode}-bert-{picl_cnt}+{icl_cnt}-shot.json"
    else:
        output_path = f"./data/{dataset}/{model_name}-{dataset}-{mode}-{picl_cnt}+{icl_cnt}-shot.json"
else:
    output_path = f"./data/{dataset}/{model_name}-{dataset}-{mode}-{picl_cnt}-shot.json"

model_path = f'/data1/jgc/models/{model_name}/'
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
llm = LLM(
    model=model_path, 
    tokenizer=model_path, 
    trust_remote_code=True, 
    tensor_parallel_size=2,
    gpu_memory_utilization=0.95
)
sampling_params = SamplingParams(
    temperature=0.0, # 0.6
    top_p=1.0, # 0.9
    max_tokens=128,
    skip_special_tokens=True,
)

with open(f'./data/{dataset}/train.json', 'r') as f:
    train_dataset = json.load(f)
type2entity = {}
for example in train_dataset:
    for type in example['label'].keys():
        norm_type = type.replace(' ', '-')
        if norm_type not in type2entity:
            type2entity[norm_type] = set()
        for entity in example['label'][type]:
            type2entity[norm_type].add(entity)
type2entity = {k: list(v) for k, v in type2entity.items()}

prompts = Prompt(dataset=dataset)
point_icl = PointICL(
    dataset=dataset, 
    type2entity=type2entity, 
    point_entity_cnt=picl_cnt, 
    use_bert=use_bert
)
baseline_prompt = prompts.baseline_prompt
icl_prompt1 = prompts.icl_prompt1
icl_prompt2 = prompts.icl_prompt2
picl_prompt1 = prompts.picl_prompt1
picl_prompt2 = prompts.picl_prompt2
fusion_prompt = prompts.fusion_prompt

def predict(texts, mode):
    prompts = []
    if mode == 'baseline':
        for text in texts:
            prompt = baseline_prompt + '\nInput: ' + text
            prompts.append(prompt)
    elif mode == 'icl':
        icl = ''
        for text in texts:
            examples = random.sample(train_dataset, icl_cnt)
            for example in examples:
                icl += 'Input: ' + example['text'] + '\nOutput: ' + str(example['label']).replace("'", '"') + '\n'
            prompt = icl_prompt1 + icl + icl_prompt2 + '\nInput: ' + text
            prompts.append(prompt)
            icl = ''
    elif mode == 'picl+icl':
        for text in texts:
            point_entity = point_icl.get_point_entity()
            if dataset == 'CoNLL2003':
                per = ", ".join(['"' + x + '"' for x in point_entity['per']])
                org = ", ".join(['"' + x + '"' for x in point_entity['org']])
                loc = ", ".join(['"' + x + '"' for x in point_entity['loc']])
                misc = ", ".join(['"' + x + '"' for x in point_entity['misc']])
                icl = ''
                examples = random.sample(train_dataset, icl_cnt)
                for example in examples:
                    icl += 'Input: ' + example['text'] + '\nOutput: ' + str(example['label']).replace("'", '"') + '\n'
                prompt = fusion_prompt.format(PER=per, ORG=org, LOC=loc, MISC=misc) + icl + icl_prompt2 + '\nInput: ' + text
            elif dataset == 'WNUT2017':
                person = ", ".join(['"' + x + '"' for x in point_entity['person']])
                location = ", ".join(['"' + x + '"' for x in point_entity['location']])
                corporation = ", ".join(['"' + x + '"' for x in point_entity['corporation']])
                product = ", ".join(['"' + x + '"' for x in point_entity['product']])
                creative_work = ", ".join(['"' + x + '"' for x in point_entity['creative-work']])
                group = ", ".join(['"' + x + '"' for x in point_entity['group']])
                icl = ''
                examples = random.sample(train_dataset, icl_cnt)
                for example in examples:
                    icl += 'Input: ' + example['text'] + '\nOutput: ' + str(example['label']).replace("'", '"') + '\n'
                prompt = fusion_prompt.format(
                    person=person, location=location, corporation=corporation, product=product,
                    creative_work=creative_work, group=group
                ) + icl + icl_prompt2 + '\nInput: ' + text
            elif dataset == 'ACE2004' or dataset == 'ACE2005':
                person = ", ".join(['"' + x + '"' for x in point_entity['person']])
                location = ", ".join(['"' + x + '"' for x in point_entity['location']])
                organization = ", ".join(['"' + x + '"' for x in point_entity['organization']])
                geographical_social_political = ", ".join(['"' + x + '"' for x in point_entity['geographical-social-political']])
                weapon = ", ".join(['"' + x + '"' for x in point_entity['weapon']])
                facility = ", ".join(['"' + x + '"' for x in point_entity['facility']])
                vehicle = ", ".join(['"' + x + '"' for x in point_entity['vehicle']])
                icl = ''
                examples = random.sample(train_dataset, icl_cnt)
                for example in examples:
                    icl += 'Input: ' + example['text'] + '\nOutput: ' + str(example['label']).replace("'", '"') + '\n'
                prompt = fusion_prompt.format(
                    person=person, location=location, organization=organization, 
                    geographical_social_political=geographical_social_political,
                    weapon=weapon, facility=facility, vehicle=vehicle
                ) + icl + icl_prompt2 + '\nInput: ' + text
            prompts.append(prompt)

    outputs = []
    for output in llm.generate(prompts, sampling_params, use_tqdm=False):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        outputs.append((prompt, generated_text))
    return outputs

idx = 0
final_res = []
with open(f'./data/{dataset}/test.json', 'r', encoding='utf-8') as fin:
    wdatas = json.load(fin)
    batch_data = []
    for i in range(0, len(wdatas), batch_cnt):
        batch_data.append(wdatas[i:(i + batch_cnt)])
    for wdata in tqdm(batch_data):
        texts = []
        for data in wdata:
            texts.append(data['text'])
        ret = predict(texts, mode)
        for text, (prompt, output) in zip(texts, ret):
            wdic = {}
            wdic['text'] = text
            wdic['result'] = output
            final_res.append(wdic)
        idx += 1
with open(output_path, 'a', encoding='utf-8') as fout:
    json.dump(final_res, fout, ensure_ascii=False, indent=2)
