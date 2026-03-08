from dataset.dataset_loader import load_dataset

train_loader, test_loader = load_dataset()

for images, labels in train_loader:
    print("Image batch:", images.shape)
    print("Labels:", labels.shape)
    break