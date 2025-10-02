import numpy as np
from torchvision import datasets, transforms

# Load CIFAR10 
cifar_train = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transforms.ToTensor()  # Convert images to tensors
)
cifar_test = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transforms.ToTensor()  # Convert images to tensors
)

# Split data into features and labels
X_train = np.stack([np.array(img) for img, _ in cifar_train])  # (50000, 32, 32, 3)
y_train = np.array([label for _, label in cifar_train])        # (50000,)
X_test = np.stack([np.array(img) for img, _ in cifar_test])  # (10000, 32, 32, 3)
y_test = np.array([label for _, label in cifar_test])        # (10000,)

# (N, H, W, C) = (50000, 32, 32, 3)
np.save("input/X_train.npy", X_train)
np.save("input/y_train.npy", y_train)

#(N, H, W, C) = (10000, 32, 32, 3)
np.save("input/X_test.npy", X_test)
np.save("input/y_test.npy", y_test)

print("Complete Save: input/X_train.npy, input/y_train.npy")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("Complete Save: input/X_test.npy, input/y_test.npy")
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)