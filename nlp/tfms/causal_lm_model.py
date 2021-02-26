import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, top_k_top_p_filtering

tokenizer = AutoTokenizer.from_pretrained('gpt2')
model = AutoModelForCausalLM.from_pretrained('gpt2')

sequence = 'Hugging Face is based in DUMBO, New York City, and '
input_ids = tokenizer.encode(sequence, return_tensors='pt')

# get logits of last hidden state
next_token_logits = model(input_ids).logits[:, -1, :]

# filter
filtered_next_token_logits = top_k_top_p_filtering(next_token_logits, top_k=50, top_p=1.0)

# sample
probs = F.softmax(filtered_next_token_logits, dim=-1)
next_tokens = torch.multinomial(probs, num_samples=10)[0]
for token in next_tokens:
    print(sequence + tokenizer.decode(token))
