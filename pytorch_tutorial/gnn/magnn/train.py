import argparse


def train(args):
    # TODO 训练时使用mini-batch 生成整个基于元路径的图占用太多内存
    pass


def main():
    parser = argparse.ArgumentParser(description='MAGNN')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--dataset', choices=['toy', 'DBLP'], default='toy', help='dataset')
    parser.add_argument('--epochs', type=int, default=200, help='number of training epochs')
    parser.add_argument('--num-heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--hidden-dim', type=int, default=64, help='dimension of hidden state')
    parser.add_argument('--dropout', type=float, default=0.6, help='feature and attention dropout')
    parser.add_argument('--lr', type=float, default=0.005, help='learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.001, help='weight decay')
    args = parser.parse_args()
    print(args)
    train(args)


if __name__ == '__main__':
    main()
