from transformers import pipeline

text_generator = pipeline('text-generation')
result = text_generator('As far as I am concerned, I will', max_length=50, do_sample=False)
print(result[0]['generated_text'])
