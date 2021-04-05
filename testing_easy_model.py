import torch
import torch.nn as nn
import torch.optim as optim


class mymodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 5)
        self.layer2 = nn.Conv2d(1, 5, 4, 2, 1)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.layer1(x)
        # only layer1 and act are used layer 2 is ignored so only layer1 and act's weight should be updated
        x = self.act(x)
        return x


model = mymodel()

weights = []

for param in model.parameters():  # loop the weights in the model before updating and store them
    # print(param)
    print("param size : ", param.size())
    weights.append(param.clone())

criterion = nn.BCELoss()  # criterion and optimizer setup
optimizer = optim.Adam(model.parameters(), lr=0.001)

foo = torch.randn(3, 10)  # fake input
target = torch.randn(3, 5)  # fake target

result = model(foo)  # predictions and comparison and backprop
loss = criterion(result, target)
optimizer.zero_grad()
loss.backward()
optimizer.step()

weights_after_backprop = []  # weights after backprop
for param in model.parameters():
    weights_after_backprop.append(param)  # only layer1's weight should update, layer2 is not used

for i in zip(weights, weights_after_backprop):
    print("same layer ? ", torch.equal(i[0], i[1]))

# **prints all Trues when "layer1" and "act" should be different, I have also tried to call param.detach in the loop but I got the same result.
