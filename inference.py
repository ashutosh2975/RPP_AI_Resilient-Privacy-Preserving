import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from models.apan_cnn import BrainTumorCNN

# Tumor classes
classes = ['glioma', 'meningioma', 'notumor', 'pituitary']


# -------------------------
# Simple Encryption Function
# -------------------------
def encrypt_data(data):

    key = 7.3

    ciphertext = data * key

    return ciphertext, key


# -------------------------
# Decryption Function
# -------------------------
def decrypt_data(ciphertext, key):

    plaintext = ciphertext / key

    return plaintext


# -------------------------
# Load Model
# -------------------------
model = BrainTumorCNN()
model.load_state_dict(torch.load("brain_tumor_model.pth"))
model.eval()


# -------------------------
# Image Transform
# -------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


# -------------------------
# Load Image
# -------------------------
image_path = "P:/Project/RPP_AI/dataset/Testing/meningioma/Te-aug-me_1.jpg"   # place MRI image here

image = Image.open(image_path)

input_tensor = transform(image)

input_tensor = input_tensor.unsqueeze(0)


# -------------------------
# Encrypt Input
# -------------------------
cipher_input, key = encrypt_data(input_tensor.numpy())

print("Encrypted Input (Ciphertext):")
print(cipher_input[0][0][:5][:5])


# -------------------------
# Decrypt before model
# -------------------------
decrypted_input = decrypt_data(cipher_input, key)

input_tensor = torch.tensor(decrypted_input, dtype=torch.float32)


# -------------------------
# Model Prediction
# -------------------------
with torch.no_grad():

    output = model(input_tensor)

    _, predicted = torch.max(output,1)

    class_name = classes[predicted.item()]


print("/nPrediction Vector:", output.numpy())

print("Predicted Tumor Class:", class_name)

from detect_tumor import detect_tumor

detect_tumor(image_path)