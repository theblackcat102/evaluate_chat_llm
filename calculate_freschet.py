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

tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
model = AutoModel.from_pretrained('facebook/contriever').eval().cuda()

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def contriever_embedding(text):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors='pt').to(0)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings.cpu().detach().numpy()



if __name__ == "__main__":
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

        answers = human
        reference, output = [], []
        chat_gpt_embed = []
        for answer, generated, chat in zip(answers, generated, chat_gpt):
            embeds = contriever_embedding([answer])[0]
            reference.append(embeds)
            embeds = contriever_embedding([generated])[0]
            output.append(embeds)
            embeds = contriever_embedding([chat])[0]
            chat_gpt_embed.append(embeds)

        output = np.concatenate(output)
        reference = np.concatenate(reference)
        chat_gpt_embed = np.concatenate(chat_gpt_embed)
        # we should cache this, but since this dataset is small,
        # ignore for now
        # https://github.com/w86763777/pytorch-gan-metrics/blob/master/pytorch_gan_metrics/core.py#L161
        m0 = np.mean(reference, axis=0)
        s0 = np.cov(reference, rowvar=False)
        model_fcd = calculate_frechet_inception_distance(output, m0, s0)
        chatgpt_fcd = calculate_frechet_inception_distance(chat_gpt_embed, m0, s0)


        print('model FCD: {}'.format(model_fcd))
        print('chatGPT FCD: {}'.format(chatgpt_fcd))

        m0 = np.mean(chat_gpt_embed, axis=0)
        s0 = np.cov(chat_gpt_embed, rowvar=False)
        model_fcd = calculate_frechet_inception_distance(output, m0, s0)
        print('model-chatGPT FCD: {}'.format(model_fcd))


