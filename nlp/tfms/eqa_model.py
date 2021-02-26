import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

name = 'bert-large-uncased-whole-word-masking-finetuned-squad'
tokenizer = AutoTokenizer.from_pretrained(name)
model = AutoModelForQuestionAnswering.from_pretrained(name)

text = r'''🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
TensorFlow 2.0 and PyTorch.'''
questions = [
    'How many pretrained models are available in 🤗 Transformers?',
    'What does 🤗 Transformers provide?',
    '🤗 Transformers provides interoperability between which frameworks?',
]
for question in questions:
    inputs = tokenizer(question, text, add_special_tokens=True, return_tensors='pt')
    input_ids = inputs['input_ids'].tolist()[0]

    text_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    output = model(**inputs)
    answer_start_scores, answer_end_scores = output.start_logits, output.end_logits

    # Get the most likely beginning of answer with the argmax of the score
    answer_start = torch.argmax(answer_start_scores)
    # Get the most likely end of answer with the argmax of the score
    answer_end = torch.argmax(answer_end_scores) + 1
    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(input_ids[answer_start:answer_end])
    )
    print('Question:', question)
    print('Answer:', answer)
