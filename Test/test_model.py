import torch
from models.apan_cnn import BrainTumorCNN

model = BrainTumorCNN()

x = torch.randn(1,1,128,128)

y = model(x)

print("Output shape:", y.shape)