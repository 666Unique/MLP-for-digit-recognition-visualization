# Import PyTorch core library: used for tensor operations and deep learning computation
import torch
# Import PyTorch neural network module: contains various layers, activation functions, etc.
import torch.nn as nn
# Import PyTorch optimizer module: contains optimization algorithms (e.g., Adam, SGD)
import torch.optim as optim
# Import DataLoader: used for batching and iterating over data
from torch.utils.data import DataLoader
# Import datasets and transforms from torchvision: for loading MNIST and preprocessing
from torchvision import datasets, transforms
# Import matplotlib for visualization
import matplotlib.pyplot as plt
# Import os for system operations (used here to set environment variables)
import os

# Set environment variable: resolves potential duplicate KMP library loading issue on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configure Chinese font display (for plots)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# --------------------------
# 1. Basic Configuration
# --------------------------
# Select computing device: prefer GPU (cuda) if available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Set random seed for reproducibility
torch.manual_seed(42)
# Print current device
print(f"Current device: {device}\n")


# --------------------------
# 2. Data Loading and Visualization
# --------------------------
# Define data transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),  # Convert image to tensor, normalize to [0, 1]
    transforms.Normalize(mean=(0.1307,), std=(0.3081,))  # Standardize using MNIST mean & std
])

# Load MNIST training dataset
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Load MNIST test dataset
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

# Set batch size
batch_size = 64

# Training data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True  # Shuffle data during training for better generalization
)

# Test data loader
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)

# --------------------------
# Data Visualization
# --------------------------
print("===== Data Sample Display =====")
images, labels = next(iter(train_loader))

print(f"Batch shape: images={images.shape}")
print(f"Single image shape: {images[0].shape} (1 channel, 28x28 pixels)")
print(f"Labels shape: {labels.shape}\n")

# Plot sample images
plt.figure(figsize=(15, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)

    # De-normalize image for visualization
    img = images[i].numpy().squeeze() * 0.3081 + 0.1307

    plt.imshow(img, cmap='gray')
    plt.title(f"Label: {labels[i].item()}")
    plt.axis('off')

plt.suptitle("Training Sample Examples", y=1.00)
plt.tight_layout()
plt.show()


# --------------------------
# 3. Model Definition
# --------------------------
class StudentMLP(nn.Module):
    def __init__(self):
        super().__init__()

        # Fully connected layer 1
        self.fc1 = nn.Linear(784, 128)
        # Fully connected layer 2
        self.fc2 = nn.Linear(128, 64)
        # Output layer
        self.fc3 = nn.Linear(64, 10)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Flatten the input (batch, 1, 28, 28) → (batch, 784)
        x = x.view(x.shape[0], -1)

        # Layer 1 + ReLU
        x = self.relu(self.fc1(x))
        # Layer 2 + ReLU
        x = self.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)

        return x


# Initialize model
model = StudentMLP().to(device)

print("\n===== Model Structure =====")
for name, layer in model.named_modules():
    if isinstance(layer, nn.Linear):
        print(f"Layer: {name}")
        print(f"Structure: Linear(in={layer.in_features}, out={layer.out_features})")
        print(f"Weights shape: {layer.weight.shape}")
        print(f"Bias shape: {layer.bias.shape}\n")


# --------------------------
# 4. Training Configuration
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 5


# --------------------------
# 5. Training and Testing
# --------------------------
train_losses = []
test_accs = []

for epoch in range(epochs):
    # Training mode
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    # Evaluation mode
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(train_loader)
    test_acc = 100 * correct / total

    train_losses.append(avg_loss)
    test_accs.append(test_acc)

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | Accuracy: {test_acc:.4f}%")


# --------------------------
# 6. Visualization
# --------------------------
plt.figure(figsize=(15, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_losses, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Accuracy curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), test_accs, 'r-')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Test Accuracy')

plt.tight_layout()
plt.show()


# --------------------------
# Prediction Visualization
# --------------------------
model.eval()
with torch.no_grad():
    images, labels = next(iter(test_loader))
    outputs = model(images.to(device))
    _, predicted = torch.max(outputs, 1)

plt.figure(figsize=(10, 4))
for i in range(9):
    plt.subplot(3, 3, i+1)

    img = images[i].numpy().squeeze() * 0.3081 + 0.1307
    plt.imshow(img, cmap='gray')

    plt.title(f"Pred: {predicted[i].item()}, True: {labels[i].item()}")
    plt.axis('off')

plt.suptitle("Test Predictions", y=1.00)
plt.tight_layout()
plt.show()


# --------------------------
# Save Model
# --------------------------
weight_filename = 'mnist_mlp_weights_ENG.pth'
torch.save(model.state_dict(), weight_filename)

full_path = os.path.abspath(weight_filename)
print(f"\nModel weights saved to: {full_path}")