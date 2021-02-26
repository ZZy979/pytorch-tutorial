from transformers import pipeline

qa = pipeline('question-answering')
context = '''Extractive Question Answering is the task of extracting an answer from a text given a question.
An example of a question answering dataset is the SQuAD dataset, which is entirely based on that task.
If you would like to fine-tune a model on a SQuAD task, you may leverage the examples/question-answering/run_squad.py script.'''
questions = [
    'What is extractive question answering?',
    'What is a good example of a question answering dataset?'
]
for question in questions:
    result = qa(question=question, context=context)
    print('Question:', question)
    print('Answer: {0[answer]}, score: {0[score]}, start: {0[start]}, end: {0[end]}'.format(result))
