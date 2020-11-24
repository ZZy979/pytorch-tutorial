import logging

from gensim.models import Word2Vec

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def train(corpus_file, size, save_path):
    model = Word2Vec(corpus_file=corpus_file, size=size, sg=1, iter=5)
    model.save(save_path)


if __name__ == '__main__':
    train(r'C:\Users\Dell\.dgl\aminer\aminer.txt', 128, r'.\model\aminer.model')
