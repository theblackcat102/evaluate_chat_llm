import os
import glob
import json
from tqdm import tqdm
from datasets import load_dataset
from bert_score import score


result_path = 'output_HC3_finance'

def calculate_bertscore(generated, answer):
    return score(generated, answer, model_type='microsoft/deberta-v2-xlarge-mnli')

if __name__ == "__main__":

    # gen_output = os.path.join(result_path, 'theblackcat102_galactica-1.3b-v2.jsonl')
    dataset = load_dataset("Hello-SimpleAI/HC3", 'finance')['train']
    for gen_output in glob.glob(os.path.join(result_path, '*.jsonl')):
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

        print("Against human")
        P, R, F = calculate_bertscore(generated, human)
        print("F1: {}".format(F.mean()))
        print("R : {}".format(R.mean()))
        print("P : {}".format(P.mean()))
        print("Against chatgpt")
        P, R, F = calculate_bertscore(generated, chat_gpt)
        print("F1: {}".format(F.mean()))
        print("R : {}".format(R.mean()))
        print("P : {}".format(P.mean()))


