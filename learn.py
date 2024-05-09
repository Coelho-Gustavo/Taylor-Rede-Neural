import torch
import numpy as np
import random
from build import TaylorSeries

def TargetF(x):
    return (x**4) + (2*(x**2)) + 1

samples = 300
sample_size = 20
x_range = [-np.pi, np.pi]
model = TaylorSeries()

epochs = 300

x_data = np.random.uniform(x_range[0], x_range[1], samples)
y_data = TargetF(x_data)

qnt = np.array_split(np.random.permutation(samples), sample_size)

def realLearner(lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    msel = torch.nn.MSELoss()

    model.train()
    train_loss = 0
    for i, j in enumerate(qnt):
        x = torch.Tensor([[x] for x in x_data[j]])
        y = torch.Tensor([[y] for y in y_data[j]])
        output = model(x)
        mse = msel(output, y)
        train_loss += mse.item()

        file = open('log.txt', 'a')
        file.write(f'Target = {y_data[i]}\n')
        file.write(f'X for Target: {x_data[i]}\n')
        file.write(f'Result: {model(x)}\n')
        file.close()

        optimizer.zero_grad()
        mse.backward()
        optimizer.step()
    nQnt = len(qnt)
    train_mse = train_loss / nQnt
    file = open('log.txt', 'a')
    file.write(f'Train MSE: {train_mse**(1/2)}\n\n')
    file.close()

def realTest():
    model.eval()
    test_loss = 0
    msel = torch.nn.MSELoss()

    with torch.no_grad():
        for i, j in enumerate(qnt):
            x = torch.Tensor([[x] for x in x_data[j]])
            y = torch.Tensor([[y] for y in y_data[j]])
            mse = msel(model(x), y)

            file = open('log.txt', 'a')
            file.write(f'Target = {y_data[i]}\n')
            file.write(f'X for Target: {x_data[i]}\n')
            file.write(f'Result: {model(x)}\n\n')
            file.close()

            test_loss += mse.item()

    nQnt = len(qnt)
    train_mse = test_loss / nQnt
    file = open('log.txt', 'a')
    file.write(f'Test MSE: {train_mse**(1/2)}\n-----------------\n\n')
    file.close()
    #print(model(x))

epoch = 0
while True:
    epoch += 1
    file = open('log.txt', 'a')
    file.write(f'Epoch {epoch}\n')
    file.close()
    realLearner(0.005)
    realTest()


#realLearner(TaylorSeries(), 0.001, 0.01, [-np.pi, np.pi], 300)
