import torch
import torch.nn as nn
import torch.optim as optim

# Input features [Average Goals Scored, Average Goals Conceded by Opponent]
X = torch.tensor([
    [50, 100], [100, 89], [89, 93], [93, 108],
    [108, 68], [68, 94], [94, 106], [106, 108],
    [108, 113]
], dtype=torch.float32)

# Target outputs [1 if the team is likely to win, 0 otherwise]
y = torch.tensor([[0], [1], [1], [0], [1], [1], [1], [1], [0]], dtype=torch.float32)

# Define the model using nn.Sequential
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  

# Train the model for 50 epochs
for epoch in range(50):  
    model.train()  # Set the model to training mode
    optimizer.zero_grad()  # Zero the gradients for iteration
    outputs = model(X)  # Compute predictions
    loss = criterion(outputs, y)  # Compute the loss
    loss.backward()  # Compute the gradient of the loss
    optimizer.step()  # Optimize the model parameters

# Create a new input tensor
new_input = torch.tensor([[50, 72]], dtype=torch.float32)

# Set the model to evaluation mode
model.eval()

# Disable gradient calculation for inference
with torch.no_grad():
    # Make a prediction for the new input
    prediction = model(new_input)

# Print the raw output from the model
print("Raw output:", prediction)

# Convert the probability to a binary class label
print("Prediction:", (prediction > 0.5).int().item())