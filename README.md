# CSCE643
Fall 2021 Term Project 

To download and setup the Udacity Self Driving Car Simulator, please follow the instructions in the repo:
https://github.com/udacity/self-driving-car-sim

This source code requires the dataset (CSV file and images) to be available in the "data" directory parallel to "code" directory. You may either choose to generate your own dataset using the training mode of the simulator, or use the public dataset hosted on kaggle: https://www.kaggle.com/zaynena/selfdriving-car-simulator

To train the network:
> python main.py

Batch size and epochs can be specified as command line arguments as:
> python main.py --batch_size <batch_size> --epochs <epochs>

The model checkpoints are saved in "models" directory parallel to "code" directory. Please make sure model has been trained and checkpoint is created before running the drive script.

To drive the car is autonomous mode:
> python drive.py
