import torch

batchSize = 64
inp = 1000
out = 10
hidden = 100
epochs = 100
x = torch.randn(batchSize, inp)
y = torch.randn(batchSize, out)
model = torch.nn.Sequential(
    torch.nn.Linear(inp, hidden),
    torch.nn.ReLU(),
    torch.nn.Linear(hidden, out)
)
# loss_ = torch.nn.PoissonNLLLoss()
loss_ = torch.nn.MSELoss()
learningRate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
for epoch in range(epochs):
    yPred = model(x)
    loss = loss_(yPred, y)
    print("Epoch: ", epoch, " Loss: ", loss)
    optimizer.zero_grad() # to overwrite the updated gradients
    loss.backward()
    optimizer.step()
