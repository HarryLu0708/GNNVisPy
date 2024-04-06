import torch
from model import GCN
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def main():
    pytorch_model = GCN(hidden_channels=64)
    # pytorch_model.load_state_dict(torch.load('pytorch_model.pt'))
    pytorch_model.eval()
    # dummy_input = (torch.zeros(1118, 7), torch.zeros(2, 2454, dtype=torch.long), torch.zeros(1118))
    #
    # torch.onnx.export(pytorch_model, dummy_input, 'onnx_model.onnx', opset_version = 12)
    # dummy_input = (torch.zeros(1118, 7), torch.zeros(2, 2454), torch.zeros(1118))

    dataset = TUDataset(root='data/TUDataset', name='MUTAG')
    print("num node features", dataset.num_node_features)
    dummy_input = torch.randn(4, dataset.num_node_features)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    torch.onnx.export(pytorch_model, (dummy_input, edge_index, batch), 'onnx_model.onnx', opset_version=12)

if __name__ == '__main__':
  main()