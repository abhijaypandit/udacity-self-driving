import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from tqdm import tqdm

from utils import parse_data, preprocess
from network import Network, PilotNet, JNet

class Model(nn.Module):

    def __init__(self, config, device):
        super(Model, self).__init__()

        self.config = config
        self.device = device
        self.network = Network().to(device)
        #self.network = PilotNet().to(device)
        #self.network = JNet().to(device)

        # Define loss function and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=0.0001)

    def train(self, x_train, y_train, x_valid=None, y_valid=None):
        #self.network.train()
        
        num_samples = len(x_train)
        num_batches = num_samples // self.config.batch_size

        #print("num_samples = ", num_samples)
        #print("num_batches = ", num_batches)

        train_loss = []
        valid_loss = []

        for epoch in range(1, self.config.epochs+1):
            self.network.train()
            start_time = time.time()

            # Random shuffle data
            shuffle_index = np.random.permutation(num_samples)
            curr_x_train = [x_train[idx] for idx in shuffle_index]
            curr_y_train = [y_train[idx] for idx in shuffle_index]

            for batch in range(1, num_batches+1):
                batch_x = curr_x_train[(batch-1)*self.config.batch_size : batch*self.config.batch_size]
                batch_y = curr_y_train[(batch-1)*self.config.batch_size : batch*self.config.batch_size]

                batch_x_tensor = []
                batch_y_tensor = []
                
                # Parse each training sample
                for i in range(len(batch_x)):
                    x, y = parse_data(self.config.data_dir, self.config.track, batch_x[i], batch_y[i], training=True)
                    batch_x_tensor.append(x)
                    batch_y_tensor.append(y)

                batch_x_tensor = torch.from_numpy(np.array(batch_x_tensor).astype(np.float32))
                batch_y_tensor = torch.from_numpy(np.array(batch_y_tensor).astype(np.float32))

                # Send tensor to CPU/GPU
                batch_x_tensor = batch_x_tensor.to(self.device)
                batch_y_tensor = batch_y_tensor.to(self.device)

                outputs = self.network(batch_x_tensor)
                loss = self.loss(outputs.flatten(), batch_y_tensor)

                # Take an optimizer step and backprop the loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                print('\rBatch {:d}/{:d} Loss {:.4f} '.format(batch, num_batches, loss), end="")
            
            duration = time.time() - start_time
            print('Epoch {:d} Loss {:.4f} Duration {:.2f} seconds.'.format(epoch, loss, duration))

            train_loss.append(loss.cpu().detach().numpy())

            # Save the model
            if (epoch) % self.config.save_interval == 0:
                self.save_model(epoch)

                if x_valid is not None and y_valid is not None:
                    print('Evaluating model...')
                    valid_loss.append(self.evaluate(x_valid, y_valid, epoch))

        # Plot the metrics
        #self.plot(train_loss, 'Epochs', 'Training Loss')
        #self.plot(valid_loss, 'Epochs', 'Validation Loss')
        plt.figure()
        plt.plot(train_loss, color='r', label='Training')
        plt.plot(valid_loss, color='b', label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('../track{}-loss.png'.format(self.config.track))

    def evaluate(self, x_eval, y_eval, checkpoint=None):
        self.load_model(checkpoint)

        preds = []
        truth = []

        with torch.no_grad():
            for i in range(len(x_eval)):
                x, y = parse_data(self.config.data_dir, self.config.track, x_eval[i], y_eval[i])
                x = np.array(x.reshape(-1,3,64,200)).astype(np.float32)
                y = np.array(y).astype(np.float32)

                x = torch.from_numpy(x).to(self.device)
                pred = self.network(x)

                #preds.append(prediction[0].cpu().detach().numpy())
                preds.append(pred[0])
                truth.append(y)
        
            preds = torch.tensor(preds)
            truth = torch.from_numpy(np.array(truth))

            loss = self.loss(preds, truth)
            print("Loss = {:.4f}".format(loss))

        return loss

    def predict(self, x, checkpoint=None):
        #self.load_model(checkpoint)

        x = preprocess(x)
        x = np.array(x.reshape(-1,3,64,200)).astype(np.float32)
        x = torch.from_numpy(x).to(self.device)
        pred = self.network(x)

        # Get scalar item from numpy array
        pred = pred[0].cpu().detach().numpy().item()
        return pred

    def plot(self, data, xlabel, ylabel):
        plt.figure()
        plt.plot(data)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig('../{}.png'.format(ylabel))

    def save_model(self, checkpoint):
        torch.save(self.network.state_dict(), "../models/track{}/model-{}.pth".format(self.config.track, checkpoint))
        print("Model checkpoint saved.")

    def load_model(self, checkpoint=None):
        checkpoint = self.config.epochs if checkpoint is None else checkpoint
        model = torch.load("../models/track{}/model-{}.pth".format(self.config.track, checkpoint), map_location=self.device)
        self.network.load_state_dict(model, strict=True)
        print("Model checkpoint {} loaded. ".format(checkpoint), end="")
        self.network.eval()

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = Model(None, device)
