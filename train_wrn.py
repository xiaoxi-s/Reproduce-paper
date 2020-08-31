
import torch 
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable


import load_data
import wide_resnet as wrn


def random_choose(X_train, y_train, batch_size = 256):
    print(len(X_train))
    if (batch_size >= len(X_train)):
        return X_train, y_train
    indices = torch.from_numpy(np.random.choice(len(X_train), batch_size))

    return X_train.index_select(0, indices), y_train.index_select(0, indices)


def test_wrn(model):
    total_loss = 0
    correct = 0
    size = len(X_test)
    total = 0 # num of samples
    transformer = transforms.Compose([
        transforms.ToTensor()
    ])

    testset = tv.datasets.CIFAR10(root='./data', train=False, download=True, transform=transformer)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

    calculate_loss = nn.CrossEntropyLoss()
    
    test_string = "# of batches: {}, loss: {}\n"
    print("Test 100 samples each time")

    with torch.no_grad():
        for step, (X, y) in enumerate(testloader):
            y_bar = model(X)
            loss = calculate_loss(y_bar, y)

            total_loss += loss.item()
            _, predicted = torch.max(y_bar.data, 1)

            correct += predicted.eq(y.data).cpu().sum()
            total += len(X)

            # save output
            with open("output.txt", "a") as output_file:
                output_file.write(test_string.format(step, loss))

            print(test_string.format(step, loss))

    with open("output.txt", "a") as output_file:
        output_file.write("Total loss: {}, correct: {}, total: {}, succ ratio: {}".format(total_loss, correct, total, 1.0*correct/total))

    print("Total loss: {}, correct: {}, total: {}, succ ratio: {}".format(total_loss, correct, total, 1.0*correct/total))


# training with all examples as torch.Tensor that are not modified
def naive_training(X_train, y_train, epoch, batch_size = 256, learning_rate = 0.1):
    # depth, widening factor, dropout rate from WRN paper, number of classes
    model = wrn.Wide_ResNet(16, 10, 0.3, 10)
    loss = 0
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    calculate_loss = nn.CrossEntropyLoss()

    output_string = "Numpy Array - Step {}, loss: {}\n"
    for i in range(epoch):
        optimizer.zero_grad()

        # SGD
        X_batch, y_batch = random_choose(X_train, y_train, batch_size = batch_size)
        y_bar = model(X_batch)

        loss = calculate_loss(y_bar, y_batch)
        loss.backward()

        optimizer.step()
        
        # save output
        with open("output.txt", "a") as output_file:
            output_file.write(output_string.format(i, loss))
        
        # print to thescreen also
        print(output_string.format(i, loss))

    return model


# training with Dataloader
def nt():
    model = wrn.Wide_ResNet(16, 10, 0.3, 10)
    
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    calculate_loss = nn.CrossEntropyLoss()
    
    loss = 0
    transformer = transforms.Compose([
        transforms.ToTensor()
    ])
    trainset = tv.datasets.CIFAR10(root='./data', train=True, download=True, transform=transformer)
    trainloader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=2)

    output_string = "Dataloader - Step {}, loss: {}\n"
    for step, (X, y) in enumerate(trainloader):
        optimizer.zero_grad()

        y_bar = model(X)

        loss = calculate_loss(y_bar, y)
        loss.backward()

        optimizer.step()

        # save output
        with open("output.txt", "a") as output_file:
            output_file.write(output_string.format(step, loss))

    return model


if __name__ == '__main__':
    
    X_train, y_train, X_test, y_test, label_names = load_data.load()
    
    X_train = torch.Tensor(X_train)
    y_train = torch.from_numpy(y_train)
    
    #model = naive_training(X_train, y_train, 200)
    model = nt()
    test_wrn(model)

    '''
    x, y = random_choose(X_train, y_train, 2)
    '''
