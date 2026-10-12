import argparse
import random


class Config:
    lr = 0.1
    epochs = 3


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

    def fit(self, data):
        for _ in range(self.cfg.epochs):
            random.shuffle(data)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    return parser.parse_args()


def main():
    Trainer(Config()).fit(list(range(10)))


if __name__ == "__main__":
    main()
