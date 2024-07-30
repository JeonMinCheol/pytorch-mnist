import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model import Net
from train import train, PATH
from test import test
import os
import pickle

import grpc
import protos.message_pb2_grpc as pb2_grpc
import protos.message_pb2 as pb2

class sendMessageClient(object):
    def __init__(self):
        self.host = 'localhost'
        self.server_port = 50051

        self.channel = grpc.insecure_channel(
            '{}:{}'.format(self.host, self.server_port))

        self.stub = pb2_grpc.sendParamsStub(self.channel)

    def get_url(self, message):
        message = pb2.Message(**message)
        print("Send model.")
        return self.stub.GetServerResponse(message)

if os.path.exists("./checkpoint") == False:
    os.mkdir("./checkpoint")
    

if os.path.exists("./checkpoint"):
    os.chmod("./checkpoint", 755)
    
lr = 1e-5
epoch = 100
batch = 64

train_dataset = datasets.MNIST(root='data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='data', train=False, transform=transforms.ToTensor(), download=True)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch, shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch, shuffle=False)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = Net().to(device)

loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr)

client = sendMessageClient()


for ep in range(epoch):
    print(f"EPOCH : {ep}")
    train(train_dataloader, model, optim, loss_fn, device)
    test(test_dataloader, model, loss_fn, device)
    
    with open("./checkpoint/checkpoint.pth", "rb") as file:
        content = file.read()
    
        result = client.get_url({"device":"localhost", "model" : content})