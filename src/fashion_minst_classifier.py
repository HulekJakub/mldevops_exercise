import torch
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.model_selection import train_test_split

fashion_mnist = torchvision.datasets.FashionMNIST("./", download=True, train=True)
fashion_mnist_test = torchvision.datasets.FashionMNIST("./", download=True, train=False)
# data_loader = torch.utils.data.DataLoader(imagenet_data,
#                                           batch_size=4,
#                                           shuffle=True,
#                                           num_workers=args.nThreads)
fashion_mnist.data.shape


# Split the dataset
X_train, X_val, y_train, y_val = train_test_split(
    fashion_mnist.data, fashion_mnist.targets, test_size=0.2, random_state=0
)

# Standardize the data
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = fashion_mnist_test.data / 255.0
y_test = fashion_mnist_test.targets
y_test

n_classes = len(fashion_mnist.classes)
n_classes

y_train_oh = F.one_hot(y_train, n_classes)
y_test_oh = F.one_hot(y_test, n_classes)
y_val_oh = F.one_hot(y_val, n_classes)


class FashionMnistClassifier(nn.Module):
    def __init__(self):
        super(FashionMnistClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 100)  # 8 features, 16 neurons in first hidden layer
        self.output = nn.Linear(100, n_classes)  # Output layer

    def forward(self, x):
        x = self.flatten(x)
        x = F.sigmoid(self.fc1(x))
        x = F.softmax(self.output(x), dim=-1)
        return x


model = FashionMnistClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.02)

inputs = torch.tensor(X_train, dtype=torch.float32)
labels = torch.tensor(y_train_oh, dtype=torch.float32)
val_inputs = torch.tensor(X_val, dtype=torch.float32)
val_labels = torch.tensor(y_val_oh, dtype=torch.float32)

epochs = 200
for epoch in range(epochs):
    # Forward pass
    model.train()
    outputs = model(inputs)
    loss = criterion(outputs, labels)

    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        model.eval()
        val_outputs = model(val_inputs)
        val_loss = criterion(val_outputs, val_labels)

        y_pred = torch.argmax(outputs, dim=-1)
        train_acc = torch.sum(y_pred == y_train) / inputs.shape[0]

        y_pred_val = torch.argmax(val_outputs, dim=-1)
        val_acc = torch.sum(y_pred_val == y_val) / val_inputs.shape[0]

    if (epoch + 1) % 5 == 0:
        print(
            f"Epoch [{epoch+1}/{epochs}], Acc:{train_acc.item():.4f}, ValAcc:{val_acc.item():.4f},  Loss: {loss.item():.4f}, ValLoss: {val_loss.item():.4f}"
        )

with torch.no_grad():
    y_predicted = model(torch.tensor(X_test, dtype=torch.float32))
    y_predicted_cls = torch.argmax(y_predicted, dim=-1)
    acc = torch.sum(y_predicted_cls == y_test) / X_test.shape[0]
    print(f"Accuracy: {acc:.4f}")
