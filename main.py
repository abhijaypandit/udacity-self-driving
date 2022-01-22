import argparse
import torch

from utils import load_data, train_test_split
from model import Model

def configure():
    """ Argument list and their default values.
    """
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--data_dir", type=str, default='../data', help='relative path to dataset directory')
    parser.add_argument("--track", type=int, default=1, help='select between tracks 1 and 2')

    parser.add_argument("--epochs", type=int, default=20, help='number of epochs')
    parser.add_argument("--batch_size", type=int, default=128, help='training batch size')
    parser.add_argument("--save_interval", type=int, default=1, help='model checkpoint save interval')

    return parser.parse_args()

def main():
    config = configure()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # Load the dataset
    x, y = load_data(config.data_dir, config.track)

    # Split original dataset into training, validation and test set
    x_train, y_train, x_test, y_test = train_test_split(x, y)
    x_train, y_train, x_valid, y_valid = train_test_split(x_train, y_train)

    model = Model(config, device)

    print("\nTraining model...")
    model.train(x_train, y_train, x_valid, y_valid)
    print("\nTesting model...")
    model.evaluate(x_test, y_test, config.epochs)

if __name__ == '__main__':
    main()