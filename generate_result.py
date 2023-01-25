import os
import torch
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
# from pytorch_gan_metrics.core import calculate_frechet_inception_distance

# change to finance, reddit_eli5, medicine, open_qa
subset = 'reddit_eli5'
os.makedirs('output_HC3_'+subset, exist_ok=True)

model_name = 'sanagnos/bloomz-1b6-finetuned'

# map question to different types of format we are using
model_mapping_type = {
    'sanagnos/bloomz-1b6-finetuned': 'v1',
    # same as OpenAssistant/galactica-6.7b-finetuned
    'sanagnos/galactica-6.7b-finetuned': 'v1',
    'theblackcat102/galactica-1.3b-v2': 'v2',
    'theblackcat102/galactica-1.3b-conversation-finetuned': 'v1',
    'Rallio67/custom_1.4B_512bs': 'rallio',
    'Rallio67/custom_7B_512bs': 'rallio',
    'Rallio67/custom_3B_512bs': 'rallio',
}

def prompt_text(text, method='v1'):
    if method == 'v1':
        return '<question>'+text+'<answer>'
    elif method == 'v2':
        return '<human>'+text+'<bot>'
    elif method == 'rallio':
        return 'User: '+text+'\n\nRosey: '
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
    print('unknown format', output)
    return output

print(model_name)
output_name = model_name.replace('/', '_')+'_temp_7'

tokenizer = AutoTokenizer.from_pretrained(model_name)
if 'Rallio' in model_name:
    tokenizer.add_special_tokens({'pad_token': '<|endoftext|>'})
model = AutoModelForCausalLM.from_pretrained(model_name).half().eval().cuda()


dataset = load_dataset("Hello-SimpleAI/HC3", subset)['train']

mapping = []
with open('output_HC3_'+subset+'/'+output_name+'.jsonl', 'w') as f:
    for row in tqdm(dataset, dynamic_ncols=True):
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
