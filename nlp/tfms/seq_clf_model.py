import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tokenizer = AutoTokenizer.from_pretrained('bert-base-cased-finetuned-mrpc')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-cased-finetuned-mrpc')
sentence0 = 'The company HuggingFace is based in New York City'
sentence1 = 'Apples are especially bad for your health'
sentence2 = "HuggingFace's headquarters are situated in Manhattan"
paraphrase = tokenizer(sentence0, sentence2, return_tensors='pt')
not_paraphrase = tokenizer(sentence0, sentence1, return_tensors='pt')

paraphrase_logits = model(**paraphrase).logits
not_paraphrase_logits = model(**not_paraphrase).logits
print('paraphrase_logits:', paraphrase_logits)
print('not_paraphrase_logits:', not_paraphrase_logits)

paraphrase_results = torch.softmax(paraphrase_logits, dim=1)
not_paraphrase_results = torch.softmax(not_paraphrase_logits, dim=-1)

# Should be paraphrase
print('paraphrase_results:', paraphrase_results)
# Should not be paraphrase
print('not_paraphrase_results:', not_paraphrase_results)
