from neuroevolution import NeuroEvolution
from base_net import BasicNet
import os
import torch

neu = NeuroEvolution()

# root_dir = "results/basic_net_res/weights/sgd/basicnet"
# model_names = os.listdir(root_dir)
# model_paths = []
# for i in range(len(model_names)):
#     model_paths.append(f"{root_dir}/{model_names[i]}")

model1 = BasicNet()
model1.load_state_dict(torch.load(
    "results/basic_net_res/weights/sgd/basicnet/basicnet_0.pth"))

model1_encoded = neu.encode_network(model1)

model1_decoded = neu.decode_candidate(model1_encoded, index=0)
torch.save(model1_decoded.state_dict(),
           "results/basic_net_res/weights/tests/encode_decode1.pth")
