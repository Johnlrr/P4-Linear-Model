import argparse
from linear_model import train_basic_models
from advanced_model import train_advanced_models

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['basic','advanced'], required=True)
    args = parser.parse_args()
    if args.task == 'basic':
        train_basic_models()
    else:
        train_advanced_models()