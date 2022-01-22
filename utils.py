import os
import csv
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt

from cv2 import resize, INTER_AREA
from tqdm import tqdm

def parse_data(data_dir, track, x, y, training=False):
    """ Parse the data from CSV
    """
    data_dir = data_dir = os.path.join(os.getcwd(), data_dir, 'track1' if track == 1 else 'track2')
    
    image = img.imread(os.path.join(os.getcwd(), data_dir, x))
    image = preprocess(image)
    #image = np.array(image).astype(np.float32)

    steer = float(y)

    # Add some offset to steer values for left and right images
    if 'left' in x:
        steer += 0.2
    elif 'right' in x:
        steer -= 0.2

    # Random flip of training sample
    if training is True:
        image, steer = random_flip(image, steer)

    return image, steer

def random_flip(x, y):
    """ Randomly flip the original data with 50% probability
    """
    if(np.random.randint(2) == 1):
        # Flip the image (horizontal)
        x = np.transpose(x, [1, 2, 0])
        x = np.fliplr(x)
        x = np.transpose(x, [2, 0, 1])

        # Flip the sign of steer value
        y = -y

    return x, y

def preprocess(image):
    """ Prepocessing the raw image before feeding to model
    """
    # Crop ROI
    image = image[35:135, :, :]

    # Shrink/resize image
    image = resize(image, (200, 64), interpolation = INTER_AREA)

    # Subtract the mean and divide by the standard deviation of the pixels
    mean = np.mean(image)
    std = np.std(image)
    image = (image - mean)/std

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])

    return image

def load_data(data_dir, track):
    """ Load the dataset
    """
    data_dir = os.path.join(os.getcwd(), data_dir, 'track1' if track == 1 else 'track2')

    # Read the CSV file -> ['center', 'left', 'right', 'steer', 'throttle', 'reverse', 'speed']
    with open(os.path.join(data_dir, 'driving_log.csv'), 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    images = []
    steers = []

    # Extract images and steer values
    for i in range(len(data)):
        for j in range(3):
            images.append(data[i][j])
            steers.append(data[i][3])

    assert(len(images) == len(steers))
    #print("Total samples = ", len(images))

    return images, steers

def train_test_split(x, y, train_ratio=0.8):
    """ Split the original dataset into a training and testing dataset.
    """
    split_index = int(len(x)*train_ratio)

    return x[:split_index], y[:split_index], x[split_index:], y[split_index:]

if __name__ == '__main__':
    print('Debugging utils.py')

    x, y = load_data('../data', track=1)
    print('x = ', len(x))
    print('y = ', len(y))
