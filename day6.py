import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

#Define transformations 

transformation = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5,), (0.5,))])


#load the Datasets

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transformation)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transformation)

#Create DataLoaders

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print("Train dataset size: ", len(train_dataset))
print("Test dataset size: ", len(test_dataset))

#Define the model

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = NeuralNetwork()

print(model)

#Define the Loss Functions and Optimizer

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

#Training the Model (loop)

def train_model(model,train_loader,criterion,optimizer,epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            #Zero Gradients
            optimizer.zero_grad()

            #Foward Pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            #Backward Pass and Optimize
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}") 

#train the Model


train_model(model,train_loader,criterion,optimizer)

#Evaluate the Model

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Accuracy: {100*correct/total:.2f}%")


evaluate_model(model, test_loader)

#Save the Model

torch.save(model.state_dict(), 'mnist_model.pth')

#Reload the Model

loaded_model = NeuralNetwork()
loaded_model.load_state_dict(torch.load('mnist_model.pth'))

#Evaluate the loaded Model

evaluate_model(loaded_model, test_loader)

#update Optimizer with a different learning rate

#use below code to update the optimizer with a different learning rate

# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# train_model(model, train_loader, criterion, optimizer)

# evaluate_model(model, test_loader)
