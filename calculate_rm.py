import glob
import os
import torch
import json
from tqdm import tqdm
import numpy as np
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from pytorch_gan_metrics.core import calculate_frechet_inception_distance

result_path = 'output_HC3_finance'

rm_tokenizer = AutoTokenizer.from_pretrained("OpenAssistant/reward-model-deberta-v3-large")
rm_model = AutoModelForSequenceClassification.from_pretrained("OpenAssistant/reward-model-deberta-v3-large").half().eval().cuda()

def evaluate_rm_score(prompts, generated):
    scores = []
    for question, answer in zip(prompts, generated):
        rank_inputs = rm_tokenizer(question, answer, return_tensors="pt", padding=True, max_length=512, truncation=True).to(0)
        score = rm_model(**rank_inputs).logits[0].cpu().detach()
        scores.append(score.item())
    return np.mean(scores), np.std(scores)


if __name__ == "__main__":
    dataset = load_dataset("Hello-SimpleAI/HC3", 'finance')['train']

    for fdx, gen_output in enumerate(glob.glob(os.path.join(result_path, '*.jsonl'))):
        print(os.path.basename(gen_output.split('.json')[0]))
        id2result = {}
        with open(gen_output, 'r') as f:
            for line in f:
                row = json.loads(line)
                id2result[row['id']] = row['answer']

        failed = False
        prompts = []
        generated = []
        human = []
        chat_gpt = []
        for row in dataset:
            if row['id'] not in id2result:
                failed = True
                break
            generated.append(id2result[row['id']])
            chat_gpt.append(row['chatgpt_answers'][0])
            human.append(row['human_answers'][0])
            prompts.append(row['question'][0])

        if failed or len(chat_gpt) != len(generated):
            print('result not matched {}/{}'.format(len(id2result), len(dataset)))
            continue

        if fdx == 0:
            ref_scores = evaluate_rm_score(prompts, human)
            print("Human RM score : {:.5f}({:.5f})".format(*ref_scores))
        gen_scores = evaluate_rm_score(prompts, generated)
        print("Model RM score : {:.5f}({:.5f})".format(*gen_scores))
        if fdx == 0:
            chat_scores = evaluate_rm_score(prompts, chat_gpt)
            print("ChatGPT RM score : {:.5f}({:.5f})".format(*chat_scores))

