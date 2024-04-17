import torch
import time
import json

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
        h = self.conv1(x, edge_index)
        h1 = h.relu()
        h2 = self.conv2(h1, edge_index)
        h3 = h2.relu()
        h4 = self.conv3(h3, edge_index)
        h5 = global_mean_pool(h4, batch)  # [batch_size, hidden_channels]
        h6 = F.dropout(h5, p=0.5, training=self.training)

        # final classifier
        out = F.softmax(h6)
        
        return out, [h.tolist(), h1.tolist(), h2.tolist(), h3.tolist(), h4.tolist(), h5.tolist(), h6.tolist()]

# training function for the model
def train(loader, model):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
         optimizer.zero_grad()
         out, h = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.

# testing function for the model
def test(loader, model):
     model.eval()
     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out, h = model(data.x, data.edge_index, data.batch)
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

    # Training
    for epoch in range(1, epoches):
        train(train_loader, model)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        data = {"epoch":epoch, "train_acc":train_acc, "test_acc":test_acc}
        yield f"data:{data}\n\n"
        # time.sleep(0.3)
    model_path = "../model.pth"
    torch.save(model.state_dict(), model_path)
    print("saved model!")
    # return li


'''get the structures of the GCN model'''
def model_to_json(model):
    model_dict = {}
    for name, module in model.named_modules():
        if name:  # This skips the root module
            module_type = str(module.__class__).split(".")[-1].split("'")[0]
            params = {param_name: param.size() for param_name, param in module.named_parameters()}
            model_dict[name] = {
                "type": module_type,
                "parameters": params
            }
    return json.dumps(model_dict, indent=4, default=lambda o: tuple(o))


'''function for prediction'''
def predict(model, dummy_input, edge_index, batch):
    model.eval()
    # predict
    with torch.no_grad():
        output, h = model(dummy_input, edge_index, batch)
    predict = output.argmax(dim=1)
    result = predict.tolist()
    return result, h
