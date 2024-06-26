# Install required packages.
import os
import torch
os.environ['TORCH'] = torch.__version__
print(torch.__version__)
#
# # pip install -q torch-scatter -f https://data.pyg.org/whl/torch-2.1.0.html
# # pip install -q torch-sparse -f https://data.pyg.org/whl/torch-2.1.0.html
# # pip install -q git+https://github.com/pyg-team/pytorch_geometric.git
#
#
from torch_geometric.datasets import TUDataset
#
dataset = TUDataset(root='data/TUDataset', name='MUTAG')
#
print()
print(f'Dataset: {dataset}:')
print('====================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

data = dataset[0]  # Get the first graph object.

print()
print(data)
print('=============================================================')

# Gather some statistics about the first graph.
print(f'Number of nodes: {data.num_nodes}')
print(f'Number of edges: {data.num_edges}')
print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
print(f'Has isolated nodes: {data.has_isolated_nodes()}')
print(f'Has self-loops: {data.has_self_loops()}')
print(f'Is undirected: {data.is_undirected()}')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')

from torch_geometric.loader import DataLoader

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

for step, data in enumerate(train_loader):
    print(f'Step {step + 1}:')
    print('=======')
    print(f'Number of graphs in the current batch: {data.num_graphs}')
    print(data)
    print()

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GGN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GGN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x, edge_index, batch):
        # print("Input shapes:")
        # print("x:", x.shape)
        # print("edge_index:", edge_index.shape)
        # print("batch:", batch.shape)
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x


model = GGN(hidden_channels=64)
print(model)

model = GGN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         optimizer.zero_grad()
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         # optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         # print(data.x)
         # print(data.edge_index)
         # print(data.)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.


for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

print("=====================================")
for name, param in model.state_dict().items():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")

# export model
model_path = "model.pth"
torch.save(model.state_dict(), model_path)
print("successfully save model to the designated path!")

# utility functions for the backend
# get the result
def predict(dummy_input, edge_index, batch):
    model.eval()
    # predict
    with torch.no_grad():
        output = model(dummy_input, edge_index, batch)
    predict = output.argmax(dim=1)
    result = predict.tolist()
    return result


#
# model = GCN(hidden_channels=64)
#
# # Create a dummy input tensor for the model
# dummy_input = torch.randn(4, dataset.num_node_features)
# edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
# batch = torch.tensor([0, 0, 0, 1], dtype=torch.long)
#
# # Create a dummy Data object for the model
# # dummy_data = Data(x=dummy_input, edge_index=edge_index, batch=batch)
#
# # Export the model to ONNX format
# onnx_file_path = "gcn_model.onnx"
# torch.onnx.export(model, (dummy_data.x, dummy_data.edge_index, dummy_data.batch), onnx_file_path, opset_version=12)
#
# print(f"Model successfully exported to {onnx_file_path}")

