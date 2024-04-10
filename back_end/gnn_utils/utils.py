import torch

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

# GCN model
class GCN(torch.nn.Module):
    def __init__(self, hidden_channels, dataset):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x

# training function for the model
def train(loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
         optimizer.zero_grad()
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.

# testing function for the model
def test(loader, model):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

# the general training loop function for the model
def training_loop(model, dataset, epoches):
    # Data Preparation
    torch.manual_seed(12345)
    dataset = dataset.shuffle()
    train_dataset = dataset[:150]
    test_dataset = dataset[150:]
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    li = []
    # Training
    for epoch in range(1, epoches):
        train(train_loader, model)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        d = {"epoch":epoch, "train_acc":train_acc, "test_acc":test_acc}
        li.append(d)
    model_path = "../model.pth"
    torch.save(model.state_dict(), model_path)
    return li


'''function for prediction'''
def predict(model, dummy_input, edge_index, batch):
    model.eval()
    # predict
    with torch.no_grad():
        output = model(dummy_input, edge_index, batch)
    predict = output.argmax(dim=1)
    result = predict.tolist()
    return result
