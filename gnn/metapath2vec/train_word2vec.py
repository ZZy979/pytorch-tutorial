import argparse
import logging

from gensim.models import Word2Vec

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description='Train word2vec for metapath2vec')
    parser.add_argument('--size', type=int, default=128, help='dimensionality of the word vectors')
    parser.add_argument('--workers', type=int, default=3, help='number of worker threads')
    parser.add_argument('--iter', type=int, default=10, help='number of iterations')
    parser.add_argument('corpus_file', help='path to corpus file')
    parser.add_argument('save_path', help='path to file where to save word2vec model')
    args = parser.parse_args()
    print(args)

    model = Word2Vec(
        corpus_file=args.corpus_file, size=args.size, min_count=1,
        workers=args.workers, sg=1, iter=args.iter
    )
    model.save(args.save_path)


if __name__ == '__main__':
    main()
