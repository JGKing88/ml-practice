import random
import numpy as np
import torch
from torch import nn
import matplotlib
import matplotlib.pyplot as plt

import torch.optim as optim

#set seed to be able to replicate the resutls
seed = 172
random.seed(seed)
torch.manual_seed(seed)

def generate_sin_wave_data():
    T = 20
    L = 1000
    N = 200

    # ar = np.array([[1, 1, 1, -1], [2, 2, 2, -2], [3, 3, 3, -3], [4, 4, 4, -4]])
    # ar = torch.from_numpy(ar)
    # print(ar.chunk(3, dim=1))

    x = np.empty((N, L), 'int64')
    x[:] = np.array(range(L)) + np.random.randint(-4 * T, 4 * T, N).reshape(N, 1)
    data = np.sin(x / 1.0 / T).astype('float64')
    input = torch.from_numpy(data[3:, :-1])
    # print(input.size())
    # print(input.size(0))
    # chunked = input.chunk(input.size(1), dim=1)
    # print(chunked)
    # print(chunked[0])
    return data

class Sequence(nn.Module):
    def __init__(self):
        super(Sequence, self).__init__()

        self.rnn1 = nn.LSTMCell(1, 20)
        self.rnn2 = nn.LSTMCell(20, 20)
        # self.rnn3 = nn.LSTMCell(51, 51)

        self.linear = nn.Linear(20, 1)

    def forward(self, input, future=0):
        outputs = []
        h_t = torch.zeros(input.size(0), 20, dtype=torch.double)
        c_t = torch.zeros(input.size(0), 20, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), 20, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), 20, dtype=torch.double)
        # h_t3 = torch.zeros(input.size(0), 51, dtype=torch.double)
        # c_t3 = torch.zeros(input.size(0), 51, dtype=torch.double)

        for i, input_t in enumerate(input.chunk(input.size(1), dim=1)):

            h_t, c_t = self.rnn1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            # h_t3, c_t3 = self.rnn3(h_t2, (h_t3, c_t3))

            output = self.linear(h_t2)

            
            outputs += [output]
        

        # if we should predict the future
        for i in range(future):

            h_t, c_t = self.rnn1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.rnn2(h_t, (h_t2, c_t2))
            # h_t3, c_t3 = self.rnn3(h_t2, (h_t3, c_t3))

            output = self.linear(h_t2)
            outputs += [output]

        outputs = torch.stack(outputs, 1).squeeze(2)
        return outputs


def train():
    # load data and make training set
    data = generate_sin_wave_data()
    input = torch.from_numpy(data[3:, :-1])
    target = torch.from_numpy(data[3:, 1:])
    test_input = torch.from_numpy(data[:3, :-1])
    test_target = torch.from_numpy(data[:3, 1:])

    seq = Sequence()

    seq.double()
    criterion = nn.MSELoss()
    # use LBFGS as optimizer since we can load the whole data to train
    optimizer = optim.LBFGS(seq.parameters(), lr=0.8)
    
    # begin to train
    for i in range(5):
        print('STEP: ', i)

        def closure():
            optimizer.zero_grad()
            out = seq(input)
            loss = criterion(out, target)
            print('loss:', loss.item())
            loss.backward()
            return loss

        optimizer.step(closure)
        
        # begin to predict, no need to track gradient here
        with torch.no_grad():
            future = 1000
            pred = seq(test_input, future=future)
            loss = criterion(pred[:, :-future], test_target)
            print('test loss:', loss.item())
            y = pred.detach().numpy()
            
        # draw the result
        plt.figure(figsize=(30, 10))
        plt.title('Predict future values for time sequences\n(Dashlines are predicted values)', fontsize=30)
        plt.xlabel('x', fontsize=20)
        plt.ylabel('y', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        def draw(yi, color):
            plt.plot(np.arange(input.size(1)), yi[:input.size(1)], color, linewidth=2.0)
            plt.plot(np.arange(input.size(1), input.size(1) + future), yi[input.size(1):], color + ':', linewidth=2.0)

        draw(y[0], 'r')
        draw(y[1], 'g')
        draw(y[2], 'b')
        plt.show()


if __name__ == '__main__':
    generate_sin_wave_data()
    train()