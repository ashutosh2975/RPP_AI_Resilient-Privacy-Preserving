import torch
import torch.nn as nn
import torch.optim as optim

from dataset.dataset_loader import load_dataset
from models.apan_cnn import BrainTumorCNN


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# Load dataset
train_loader, test_loader = load_dataset()

# Load model
model = BrainTumorCNN().to(device)

# Loss
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10

for epoch in range(epochs):

    model.train()
    total_loss = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs}  Loss: {total_loss:.4f}")

# Save trained model
torch.save(model.state_dict(), "brain_tumor_model.pth")

print("Training Finished and Model Saved")