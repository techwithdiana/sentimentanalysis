"""Main file for sentiment analysis
run as: python3 -W ignore main.py --max_length 256 --transformer sentence-transformers/all-MiniLM-L6-v2 -lr 0.00001 --n_epochs 10 --batch_size 32
"""

import os
import argparse
from dataset import Dataset 
from pipeline import Pipeline


def main(args):
    pip = Pipeline(
        transformer=args.transformer,
        lr=args.lr)
    trn_dataset = Dataset(
        os.path.join(os.getcwd(), 'data', 'train'),
        max_length=args.max_length)
    tst_dataset = Dataset(
        os.path.join(os.getcwd(), 'data', 'test'),
        max_length=args.max_length)
    pip.train(
        trn_dataset=trn_dataset,
        val_dataset=tst_dataset,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, help="Number of epochs", default=10)
    parser.add_argument("--max_length", type=int, help="Max length", default=128)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("-lr", type=float, help="learning rate", default=0.00002)
    parser.add_argument("--transformer", type=str, help="LLM Model", default='google/bert_uncased_L-2_H-128_A-2')
    args = parser.parse_args()
    main(args)
