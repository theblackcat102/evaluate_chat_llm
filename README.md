# evaluate_chat_llm


A proof of concept on some ideas for evaluating chat model

Some challenges for generation based model

- traditional stat based n-gram model such as ROUGE, BLEU doesn't work well if the text has the same meaning but doesn't share any words with the answer

- penalty for short answer

Our problem typical has two challenges:

- no reference answer or ground truth

- even if we have ground truth, each LLMs may have different tone/style but correct answer


Some possible solutions:

- Use FID like score for text. 
    
    - IRL we can have two different topics with human/ good teacher model answer and model generated text, cause its just a fancy metrics for distribution similarity

- BertScore does have good correlation to human preference and easy to use package

- Reward model from human ranking feedback

    - Not ideal for model which train on RLHF on that reward model


Related readings:

- [A Comprehensive Assessment of Dialog Evaluation Metrics](https://arxiv.org/pdf/2106.03706.pdf)

- [Bert score github](https://github.com/Tiiiger/bert_score)

- [Frechet Inception Distance (FID) - wikipedia](https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance)

- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/pdf/2210.10760.pdf)
