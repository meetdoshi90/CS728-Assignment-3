from datasets import load_dataset
import torch
from transformers import AutoTokenizer, XGLMForCausalLM, LlamaForCausalLM
import json
import random
from copy import deepcopy
from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
from torchmetrics import Accuracy, F1Score
import numpy as np

mapper = {}

accuracy = Accuracy(task="multiclass", num_classes=3)
f1 = F1Score(task="multiclass", num_classes=3)


with open('mapper.json') as f:
    mapper = json.load(f)

print(mapper)

lang_1 = 'fr'
lang_2 = 'en'
k=5
SEED=42
random.seed(SEED)
model_name = "/raid/nlp/models/llama-2-7b-chat-hf/"

device = 'cuda:0'

data_1 = load_dataset('xnli',name=lang_1,split='test')
data_2 = load_dataset('xnli',name=lang_2,split='test')
print(len(data_1),len(data_2))


tokenizer = AutoTokenizer.from_pretrained(model_name)
if model_name=="facebook/xglm-564M":
    model = XGLMForCausalLM.from_pretrained("facebook/xglm-564M").to(device)
elif model_name=="/raid/nlp/models/llama-2-7b-chat-hf/":
    model = LlamaForCausalLM.from_pretrained("/raid/nlp/models/llama-2-7b-chat-hf/").to(device)
else:
    raise Exception("Model not found")


def sample_k_indexes(lst_len, k, cur_idx):
    indexes = []
    while len(indexes) < k:
        index = random.randint(0, lst_len - 1)
        if index not in indexes and index!=cur_idx:
            indexes.append(index)
    return indexes


inputs = tokenizer("Hello, my dog is cute", return_tensors="pt").to(device)
print(inputs)
print(tokenizer.padding_side)
print(tokenizer.bos_token_id)
print(tokenizer.eos_token_id)
print(tokenizer.pad_token_id)
print(tokenizer.sep_token_id)
print(tokenizer.vocab_size)
outputs = model(**inputs, labels=inputs["input_ids"])

def tokenize_prompt(example):
    #print(example)
    template = f'Premise: {example["premise"]} Hypothesis: {example["hypothesis"]} Label: {mapper[lang_1][str(example["label"])]}\n'
    #print(template)
    out =  tokenizer(template, return_tensors='pt')
    #print('Prompt',out.input_ids)
    return out


def tokenize_input(example):
    #print(example)
    #print('Premise', example['premise'])
    #print('Hypothesis', example['hypothesis'])
    template = 'Premise: {premise} Hypothesis: {hypothesis} Label:'.format(premise=example['premise'],hypothesis=example['hypothesis'])
    #print(template)
    out = tokenizer(template, return_tensors='pt')
    #print('Input',out.input_ids)
    return out


actuals = []
preds = []

for i in tqdm(range(500)):
    indexes = sample_k_indexes(len(data_1),k,i)
    #print(indexes)
    cur_input = {} 
    for idx in indexes:
        example = data_1[idx]
        tokenized_example = tokenize_prompt(example)
        if cur_input== {}:
            cur_input['input_ids'] = tokenized_example.input_ids
            cur_input['attention_mask'] = tokenized_example.attention_mask
        else: 
            cur_input['input_ids'] = torch.cat((cur_input['input_ids'][0,:], tokenized_example.input_ids[0,:])).view(1,-1)
            cur_input['attention_mask'] = torch.cat((cur_input['attention_mask'][0,:], tokenized_example.attention_mask[0,:])).view(1,-1)
    example = data_2[i] 
    actuals.append(example['label'])
    tokenized_example = tokenize_input(example)
    cur_input['input_ids'] = torch.cat((cur_input['input_ids'][0,:], tokenized_example.input_ids[0,:])).view(1,-1)
    cur_input['attention_mask'] = torch.cat((cur_input['attention_mask'][0,:], tokenized_example.attention_mask[0,:])).view(1,-1)
    #print('Cur input',cur_input)
    labels = torch.zeros_like(cur_input['input_ids'], dtype=torch.long) - 100
    cur_input = BatchEncoding(cur_input)
    
    cur_preds = []
    for cur_label_idx in ["0","1","2"]:
        cur_label = mapper[lang_2][cur_label_idx]
        #print(cur_label)
        tokenized_label = tokenizer(cur_label,return_tensors='pt')
        #print('$'*10)
        temp = {}
        temp['input_ids'] = torch.cat((cur_input.input_ids[0,:], tokenized_label.input_ids[0,1:])).view(1,-1)
        temp['attention_mask'] = torch.ones_like(temp['input_ids'])
        #temp['device'] = torch.device(device)
        temp_labels = torch.cat((labels[0,:-1], tokenized_label.input_ids[0,1:], torch.tensor([-100]))).to(device).view(1,-1).to(device)
        temp = BatchEncoding(temp)
        #print(temp)
        #print(type(temp))
        temp.to(device)
        #print(temp)
        #print(type(temp))
        print(temp_labels)
        #print(temp_labels.shape, temp.input_ids.shape, temp.attention_mask.shape)
        #print(type(temp))
        #print(temp.input_ids.device,temp.attention_mask.device,temp_labels.device, model.device)
        print(tokenizer.decode(temp.input_ids.tolist()[0]))
        print(tokenizer.decode(temp_labels.tolist()[0], skip_special_tokens=True))
        print('\n')
        assert temp_labels.shape==temp.input_ids.shape 
        outputs = model(**temp, labels=temp_labels)
        #print(outputs.logits.shape)
        #print(outputs.loss)
        cur_preds.append(outputs.loss.item())
    preds.append(torch.argmin(torch.tensor(cur_preds)))
    #print(cur_preds)

preds = torch.tensor(preds)
actuals = torch.tensor(actuals)

print(accuracy(preds,actuals))
print(f1(preds,actuals))