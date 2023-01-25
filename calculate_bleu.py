import os
import glob
import json
from nltk.translate.bleu_score import corpus_bleu
from datasets import load_dataset
result_path = 'output_HC3_finance'

if __name__ == "__main__":
    dataset = load_dataset("Hello-SimpleAI/HC3", 'finance')['train']

    for fdx, gen_output in enumerate(glob.glob(os.path.join(result_path, '*.jsonl'))):
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
        
        references = []
        model_hypo = []
        chat_hypo = []
        for answer, generated, chat in zip(human, generated, chat_gpt):
            answer_token = answer.split(' ')
            chat_token = chat.split(' ')
            generated_token = generated.split(' ')
            chat_hypo.append(chat_token)
            model_hypo.append(generated_token)
            references.append([answer_token])

        if fdx == 0:
            chat_bleu = corpus_bleu(references, chat_hypo)
            print('ChatGPT BLEU {}'.format(chat_bleu))
        model_bleu = corpus_bleu(references, model_hypo)
        print(os.path.basename(gen_output.split('.json')[0]), 'BLEU {}'.format(model_bleu))
