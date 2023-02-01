import os
import torch
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
# from pytorch_gan_metrics.core import calculate_frechet_inception_distance

subset = 'finance'
os.makedirs('output_HC3_'+subset, exist_ok=True)

model_name = 'EleutherAI/pythia-12b-deduped-base-finetuned/checkpoint-1000'
model_name = 'Rallio67/rosey_12B_instruct_alpha'
model_name = 'Rallio67/chip_20B_instruct_alpha'

model_mapping_type = {
    'EleutherAI/pythia-12b-deduped-base-finetuned/checkpoint-1000': 'v2',
    'sanagnos/bloomz-1b6-finetuned': 'v1',
    'facebook/galactica-6.7b-base-finetuned/checkpoint-7500': 'v2',
    # same as OpenAssistant/galactica-6.7b-finetuned
    'sanagnos/galactica-6.7b-finetuned': 'v1',
    'theblackcat102/galactica-1.3b-v2': 'v2',
    'theblackcat102/galactica-1.3b-conversation-finetuned': 'v1',
    'Rallio67/custom_1.4B_512bs': 'rallio',
    'Rallio67/custom_7B_512bs': 'rallio',
    'Rallio67/custom_3B_512bs': 'rallio',
    'Rallio67/rosey_12B_instruct_alpha': 'rallio',
    'Rallio67/chip_20B_instruct_alpha': 'rallio2', # for different name
}

def prompt_text(text, method='v1'):
    if method == 'v1':
        return '<question>'+text+'<answer>'
    elif method == 'v2':
        return '<human>'+text+'<bot>'
    elif method == 'rallio':
        return 'User: '+text+'\n\nRosey: '
    elif method == 'rallio2':
        return 'User: '+text+'\n\Chip: '
    else:
        return text

    
def process_output(output):
    if '<human>' in output: # human-bot
        question, answer = output.split('<bot>', maxsplit=1)
        answer = answer.split('</s>')[0].replace('<|endoftext|>', '').lstrip().split('<bot>')[0]
        return answer
    elif '<question>' in output: # question-answer
        question, answer = output.split('<answer>', maxsplit=1)
        answer = answer.split('</s>')[0].replace('<|endoftext|>', '').lstrip().split('<question>')[0]
        return answer
    elif 'Rosey:' in output:
        question, answer = output.split('Rosey:', maxsplit=1)
        answer = answer.split('<|endoftext|>')[0]
        return answer
    elif 'Chip:' in output:
        question, answer = output.split('Chip:', maxsplit=1)
        answer = answer.split('<|endoftext|>')[0]
        return answer
    print('unknown format', output)
    return output

print(model_name)
output_name = model_name.replace('/', '_')+'_temp_7'

tokenizer = AutoTokenizer.from_pretrained(model_name)
if 'Rallio' in model_name:
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().cuda()


dataset = load_dataset("Hello-SimpleAI/HC3", subset)['train']
output_filename = 'output_HC3_'+subset+'/'+output_name+'.jsonl'
added_id = set()
if os.path.exists(output_filename):
    # resume from previous run
    with open(output_filename, 'r') as f:
        for line in f:
            row = json.loads(line)
            added_id.add(row['id'])

mapping = []
with open(output_filename, 'a') as f:
    for row in tqdm(dataset, dynamic_ncols=True):
        if row['id'] in added_id:
            continue

        input_text = prompt_text(row['question'], method=model_mapping_type[model_name])
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(0)
        if 'token_type_ids' in inputs:
            inputs.pop('token_type_ids')
        if 'Rallio' in model_name:
            outputs = model.generate(**inputs, pad_token_id=tokenizer.eos_token_id, max_length=512, early_stopping=True, do_sample=True, temperature=0.7)
        else:
            outputs = model.generate(**inputs, max_length=512, early_stopping=True, temperature=0.7, do_sample=True)
        output = tokenizer.decode(outputs[0], truncate_before_pattern=[r"\n\n^#", "^'''", "\n\n\n"])
        output_row = {'question': row['question'], 'answer': process_output(output), 'id': row['id']}
        f.write(json.dumps(output_row)+'\n')
        f.flush()
