import sys
from pathlib import Path

current_file_path = Path(__file__).resolve()
parent_dir = current_file_path.parent.parent
sys.path.append(str(parent_dir))

import torch
from torch_geometric.datasets import TUDataset
from flask import Flask, jsonify
from python_models.model import train, test, train_loader, test_loader, GCN

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

app = Flask(__name__)

'''get the dataset stats'''
@app.route("/get_dataset_stats")
def get_data_stats():
    num_graphs = len(dataset)
    num_features = dataset.num_features
    num_classes = dataset.num_classes
    d = {"num_graphs":num_graphs, "num_features":num_features, "num_classes":num_classes}
    return jsonify(d)

'''get the number nth graph data from the dataset'''
@app.route("/get_nth_graph_data/<idx>")
def get_nth_graph_data(idx):
    idx = int(idx)
    data = dataset[idx]
    n_nodes = data.num_nodes
    n_edges = data.num_edges
    avg_node_degree = data.num_edges / data.num_nodes
    has_isolated_nodes = data.has_isolated_nodes()
    has_self_loop = data.has_self_loops()
    is_undirected = data.is_undirected()
    d = {
        "num_nodes":n_nodes,
        "num_edges":n_edges,
        "avg_node_degree":avg_node_degree,
        "has_isolated_nodes":has_isolated_nodes,
        "has_self_loop":has_self_loop,
        "is_undirected":is_undirected
    }
    return jsonify(d)

'''return the final accuracy rates for both training set and testing set'''
@app.route("/get_train_result")
def get_result():
    for epoch in range(1, 171):
        train()
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        # print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    d = {"train_acc":train_acc, "test_acc":test_acc}
    return jsonify(d)

'''get the output of GCN and return it'''
@app.route("/get_output")
def get_output():
    # init the model
    gcn = GCN(hidden_channels=64)
    # get the data for input(some dummy data)
    data = None
    dummy_input = torch.randn(4, dataset.num_node_features)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1], dtype=torch.long)
    print("=======================================")
    print("dummy_input:",dummy_input,"edge_index:",edge_index,"batch:",batch)
    gcn.eval()
    # predict
    with torch.no_grad():
        output = gcn(dummy_input, edge_index, batch)
    predict = output.argmax(dim=1)
    result = predict.tolist()
    return jsonify(result)

''''return the weights of the GCN model'''
@app.route("/get_weights")
def get_weights():
    model = GCN(hidden_channels=64)
    result = []
    for name, param in model.state_dict().items():
        # print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
        d = {"layer":name, "size":param.size(), "values":param[:2].tolist()}
        result.append(d)
    return jsonify(result)



if __name__ == "__main__":
    # app.run(debug=True)
    # for server hosting
    app.run(host='0.0.0.0', port=5000)