import torch
from dataset.dataset_loader import load_dataset
from models.apan_cnn import BrainTumorCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_loader, test_loader = load_dataset()

model = BrainTumorCNN().to(device)
model.load_state_dict(torch.load("brain_tumor_model.pth"))
model.eval()

correct = 0
total = 0

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predicted = torch.max(outputs,1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total

print("Model Accuracy:", accuracy)