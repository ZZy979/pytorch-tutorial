from pprint import pprint

from transformers import pipeline

fill_mask = pipeline('fill-mask')
print(fill_mask.tokenizer.mask_token)
masked_seq = f'HuggingFace is creating a {fill_mask.tokenizer.mask_token}' \
             f' that the community uses to solve NLP tasks.'
print(masked_seq)
pprint(fill_mask(masked_seq))
