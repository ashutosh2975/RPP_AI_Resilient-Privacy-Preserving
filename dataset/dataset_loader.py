import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dataset.dataset import get_dataset_path


def load_dataset(batch_size=32):

    training_path, testing_path = get_dataset_path()

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.ImageFolder(
        root=training_path,
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=testing_path,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    print("Classes:", train_dataset.classes)
    print("Train Images:", len(train_dataset))
    print("Test Images:", len(test_dataset))

    return train_loader, test_loader