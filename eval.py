from dataset import train_loader, test_loader
from differential_evolution import Candidate
import torch
import torch.nn as nn
from base_net import BasicNet
import os
import hyperparams as hyperparams

root_dir = "/home/iec/abhimanyu/etc/splitnet/results/basic_net_res"
model_names = os.listdir(f"{root_dir}/weights")
model_paths = []
for i in range(len(model_names)):
    model_paths.append(f"{root_dir}/weights/{model_names[i]}")

network = BasicNet()
network.load_state_dict(torch.load(model_paths[0]))

device = "cuda:0"
neural_cost_func = nn.CrossEntropyLoss()

network.to(device)
network.eval()
with torch.no_grad():
    train_acc = 0
    loss = 0
    for b, (X_train, y_train) in enumerate(train_loader):
        # run model, get loss
        # flatten and evaluate
        # print("X_shape:", X_train.shape)
        input_ = X_train.view(hyperparams.batch_size, -1).to(device)
        y_train = y_train.to(device)
        # print("inp shape:", input_.shape)
        y_pred = network.forward(input_)
        loss += neural_cost_func(y_pred.to(device),
                                 y_train.to(device)).data.item()
        # gen_train_loss.append(loss)

        predicted = torch.max(y_pred.data, 1)[1]

        train_acc += (predicted == y_train).sum()
    # print("Train acc:", (train_acc)/(b*hyperparams.batch_size))
    # gen_train_loss_total = sum(gen_train_loss)
    loss /= len(train_loader)

print(loss)
print(train_acc/len(train_loader))
