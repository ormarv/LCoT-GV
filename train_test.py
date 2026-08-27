#!/usr/bin/env python3
from argparse import ArgumentParser
import os
import networkx as nx
import json
import torch
from typing import List, Dict, Tuple
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import numpy as np
from scipy.sparse import coo_matrix
from gat_models import GAT, TextGAT, MixedGAT
from torch_geometric.explain import Explainer, GNNExplainer

def customJsonDecoder(object):
    print(f"Type of object: {type(object)}")
    if isinstance(object,dict):
        new_obj = {}
        for key, value in object.items():
            try:
                int_key = int(key)
            except ValueError:
                int_key = key
            new_obj[int_key] = value
        return new_obj
    return object

def load_train_eval_graphs_with_full_features(graphs_dir:str):
    train_graphs = []
    eval_graphs = []
    for file in os.listdir(graphs_dir):
        with open(os.path.join(graphs_dir,file),"r") as f:
            lines = f.readlines()
            if "train" in file:
                for line in lines:
                    sample = json.loads(line, object_hook=customJsonDecoder)
                    train_graphs.append((nx.from_dict_of_dicts(sample["graph"]),torch.tensor(sample["features"]),sample["label"], torch.tensor(sample["embeddings"])))
            elif "eval" in file:
                for line in lines:
                    sample = json.loads(line, object_hook=customJsonDecoder)
                    eval_graphs.append((nx.from_dict_of_dicts(sample["graph"]),torch.tensor(sample["features"]),sample["label"], torch.tensor(sample["embeddings"])))
    return train_graphs, eval_graphs

def normalize_node_index_per_graph(
    features_list: List[torch.Tensor], 
    wanted_features: Dict[str, int]
) -> List[torch.Tensor]:
    """
    Applies Min-Max scaling to the 'node_index' feature at the graph level.
    Scales the index relative to the size of the specific graph [0, 1].
    """
    print(f"The train_features passed to normalize_node_index_per_graph: {features_list}")
    if 'node_index' not in wanted_features:
        return features_list
    
    idx = wanted_features['node_index']
    normalized_features = []
    
    for f in features_list:
        f_norm = f.clone()
        num_nodes = f_norm.shape[0]
        
        if num_nodes > 1:
            f_norm[:, idx] = f_norm[:, idx] / (num_nodes - 1)
        else:
            f_norm[:, idx] = 0.0 
            
        normalized_features.append(f_norm)
        
    return normalized_features


def normalize_dataset_features(train_features:List[torch.Tensor], eval_features:List[torch.Tensor], wanted_features:Dict[str,int],do_eval:bool=True)->Tuple[List[torch.Tensor],List[torch.Tensor]]:
    log_columns = [wanted_features[k] for k in ['nb_parents','nb_children'] if k in wanted_features]
    z_columns = [wanted_features[k] for k in ['nb_nodes_per_depth','nb_words_before'] if k in wanted_features]

    norm_train_features = [f.clone() for f in train_features]
    if do_eval:
        norm_eval_features = [f.clone() for f in eval_features]
    else:
        norm_eval_features = []

    if log_columns:
        for f_list in [norm_train_features, norm_eval_features]:
            for features in f_list:
                features[:, log_columns] = torch.log1p(features[:,log_columns])

    all_z_columns = list(set(log_columns + z_columns))
    if not all_z_columns:
        print("Could not normalize the features.")
        return norm_train_features, norm_eval_features
    
    concatenated_train_nodes = torch.cat(norm_train_features, dim=0)
    features_mean = concatenated_train_nodes[:, all_z_columns].mean(dim=0)
    features_std = concatenated_train_nodes[:, all_z_columns].std(dim=0)
    print(f"Means: {features_mean}, standard deviations: {features_std}")
    # Replace the 0s to avoid division by 0
    features_std[features_std == 0] = 1.0

    for f_list in [norm_train_features, norm_eval_features]:
        for features in f_list:
            features[:, all_z_columns] = (features[:, all_z_columns]  - features_mean) / features_std

    return norm_train_features, norm_eval_features, features_mean, features_std

def normalize_test_features(
    test_features: List[torch.Tensor], 
    wanted_features: Dict[str, int], 
    features_mean: torch.Tensor, 
    features_std: torch.Tensor
) -> List[torch.Tensor]:
    """
    Normalizes test features using the mean and standard deviation 
    computed from the training dataset to prevent data leakage.
    """
    log_columns = [wanted_features[k] for k in ['nb_parents','nb_children'] if k in wanted_features]
    z_columns = [wanted_features[k] for k in ['nb_nodes_per_depth','nb_words_before'] if k in wanted_features]

    norm_test_features = [f.clone() for f in test_features]

    # Apply the same log transformation
    if log_columns:
        for features in norm_test_features:
            features[:, log_columns] = torch.log1p(features[:, log_columns])

    all_z_columns = list(set(log_columns + z_columns))
    if not all_z_columns:
        print("No columns to normalize.")
        return norm_test_features
    
    # Apply Z-score normalization using the TRAINING mean and std
    for features in norm_test_features:
        features[:, all_z_columns] = (features[:, all_z_columns] - features_mean) / features_std

    return norm_test_features

def restrict_features(train_features:List[List[float]],wanted_features:Dict[int,str])->Tuple[List[float]]:
    """
    Restricts the training features to the elements selected by the user. 

    Args:
        train_features (List[List[float]]): The complete set of node features.
        wanted_features (Dict[int,str]): The index and name of the features selected by the user.

    Returns:
        Tuple[List[float]]
    """
    restricted_features = []
    for graph_features in train_features:
        restricted_features.append(torch.tensor([[node_features[i] for _,i in wanted_features.items()] for node_features in graph_features]))
    return tuple(restricted_features)

def get_edge_index(graph:nx.DiGraph)->torch.Tensor:
    """
    Create the edge index of a graph.

    Args:
        graph (nx.DiGraph)

    Returns:
        torch.Tensor
    """
    adjacency_matrix = nx.to_numpy_array(graph)
    coo = coo_matrix(adjacency_matrix)
    edge_index = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.long)
    return edge_index

def get_edge_attr(graph: nx.DiGraph, edge_index: torch.Tensor)->torch.Tensor:
    """
    Creates a one-hot encoding for each edge based on its label.

    Args:
        graph (nx.DiGraph)
        edge_index (torch.Tensor)

    Returns:
        torch.Tensor
    """
    edges = edge_index.t().tolist()
    print(f"The edge index: {edge_index}")
    print(f"Edges (edge_index.t().tolist(): {edges})")
    print(f"nx.to_dict_of_dicts(graph): {nx.to_dict_of_dicts(graph)}")
    print(f"graph.edges: {graph.edges}")
    edge_attr = []
    for u, v in edges:
        rel = graph.edges[u, v].get("relation", 0)
        if rel == 1:
            edge_attr.append([0.0, 1.0, 0.0])
        elif rel == 2:
            edge_attr.append([0.0, 0.0, 1.0])
        else:
            edge_attr.append([1.0, 0.0, 0.0])

    if len(edge_attr) == 0:
        return torch.empty((0, 3), dtype=torch.float)
    
    return torch.tensor(edge_attr, dtype=torch.float)

def build_dataloader(all_features:List[torch.Tensor], graphs:List[nx.DiGraph], labels:List[int],batch_size:int=32)->DataLoader:
    """
    Build a training or evaluation dataloader given a graph and its features.

    Args:
        all_features (List[torch.Tensor]): The complete set of node features for all graphs.
        graphs (List[nx.DiGraph]): The list of graphs.
        labels (List[int]): The list of labels.
        batch_size (int): The batch size.

    Returns:
        DataLoader

    """
    print("Inside build_dataloader")
    iterator = zip(graphs, all_features, labels)
    datas = []
    for graph, features, label in iterator:
        edge_index = get_edge_index(graph)
        edge_attr = get_edge_attr(graph, edge_index)
       
        print(f"Testing the contents of a Data object before the loader: {Data(x=features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([int(label)], dtype=torch.long)).to_dict()}")
        
        datas.append(Data(x=features, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([int(label)], dtype=torch.long)))
        
    loader = DataLoader(datas, batch_size=batch_size, shuffle=True)
    return loader

def train(train_dataloader:DataLoader, val_loader:DataLoader, trained_model_path:str, in_channels:int|Tuple[int,int], out_channels:int, hidden:int, model_dir:str, architecture:str, metadata_size:int, epochs:int=100, lr=5e-4,decay=1e-2):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)
    if architecture=="text":
        model = TextGAT(in_channels=in_channels, out_channels=out_channels, device=device, hidden=hidden).to(device)
    elif architecture=="mixed":
        assert len(in_channels)==2
        print(f"text_dim,features_dim  (i.e. in_channels): {in_channels}")
        model = MixedGAT(text_dim=in_channels[0], features_dim=in_channels[1], out_channels=out_channels, metadata_size=metadata_size, device=device, hidden=hidden).to(device)
        print(model)
    else:
        model = GAT(in_channels=in_channels, out_channels=out_channels, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
    best_evaluation_accuracy = 0
    for epoch in range(epochs):
        print(f"-------------------------------EPOCH N°{epoch}-------------------------------")
        print("    Training")
        model.train()
        loss_all = 0
        total_correct_train = 0
        total_train = 0
        for j, data in enumerate(train_dataloader):
            data = data.to(device)
            optimizer.zero_grad()
            if architecture == "mixed":
                output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            else:
                output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            prediction = output.argmax(dim=1)
            correct = int((prediction == data.y).sum())
            acc = correct/len(data.y)
            total_correct_train += correct
            total_train +=len(data.y)
            loss = torch.nn.functional.nll_loss(output, data.y)
            loss_all += loss.item()
            loss.backward()
            optimizer.step()
            print(f"    Batch {j}. Loss: {loss.item()}. Accuracy: {acc}")
        avg_train_loss = loss_all/len(train_dataloader.dataset)
        print(f"Training loss: {avg_train_loss}")
        print("\n")
        print("    Evaluation")
        model.eval()
        total_correct_eval = 0
        total_eval = 0
        loss_eval = 0
        with torch.no_grad():
            for j, data in enumerate(val_loader):
                data = data.to(device)
                if architecture == "mixed":
                    output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                else:
                    output = model(data.x, data.edge_index, data.edge_attr, data.batch)
                loss = torch.nn.functional.nll_loss(output, data.y)
                loss_eval += loss.item()
                prediction = output.argmax(dim=1)
                correct = int((prediction == data.y).sum())
                acc = correct/len(data.y)
                print(f"    Batch {j}. Loss: {loss.item()}. Accuracy: {acc}")
                total_correct_eval += correct
                total_eval += len(data.y)
        avg_loss_eval = loss_eval/len(val_loader.dataset)
        print(f"Evaluation loss: {avg_loss_eval}")
        evaluation_accuracy = total_correct_eval/total_eval
        print(f"Evaluation accuracy: {evaluation_accuracy}")
        if evaluation_accuracy > best_evaluation_accuracy:
            best_evaluation_accuracy = evaluation_accuracy
            torch.save(model.state_dict(), trained_model_path)
        print("\n\n")

        # We save a checkpoint of the model
        filepath = os.path.join(model_dir, f"epoch_{epoch}.pth")
        torch.save(model.state_dict(), filepath)
    print(f"Best evaluation accuracy: {best_evaluation_accuracy}")
    if architecture=="text":
        best_model = TextGAT(in_channels=in_channels, out_channels=out_channels, device=device, hidden=hidden).to(device)
    elif architecture=="mixed":
        best_model = MixedGAT(text_dim=in_channels[0], features_dim=in_channels[1], out_channels=out_channels, metadata_size=metadata_size, device=device, hidden=hidden).to(device)
    else:
        best_model = GAT(in_channels=in_channels, out_channels=out_channels, hidden=hidden).to(device)
    best_model.load_state_dict(torch.load(trained_model_path))
    return best_model

def test(test_dataloader:DataLoader, model:torch.nn.Module)->float:
    """
    The test loop for the GAT model.

    Args:
        test_dataloader (DataLoader): The dataloader containing the test data.
        model (torch.nn.Module): The trained model to test.

    Returns:
        float
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device', device)
    model.to(device)
    total_correct = 0
    total = 0
    prediction_average = 0
    all_predictions = []
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader):
            data = data.to(device)
            output = model(data.x, data.edge_index, data.edge_attr, data.batch)
            predictions = output.argmax(dim=1)
            prediction_average += int(predictions.sum())
            correct = int((predictions==data.y).sum())
            acc = correct/len(data.y)
            print(f"    Batch {i}. Accuracy: {acc}")
            total_correct += correct
            total += int(len(data.y))
            all_predictions.extend(predictions.cpu().tolist())
    avg_accuracy = total_correct/total
    return avg_accuracy

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("actions",nargs='+',type=str,choices=["train","test"],help="Whether to train or test the model, or both.")
    parser.add_argument("-v","--verbose",action="store_true",help="Print extra information about the process.")
    parser.add_argument("-g","--graphs-dir",type=str,default="data/graphs/",help="Directory where the data files containing the graphs and their features are stored.")
    parser.add_argument("-b","--batch-size",type=str,default=32)
    parser.add_argument("-e","--epochs",type=int,default=100)
    parser.add_argument("-l","--learning-rate",type=float,default=1e-3)
    parser.add_argument("-w","--weight-decay",type=float,default=1e-2)
    parser.add_argument("-f", "--wanted-features",type=str,nargs='+',choices=['nb_parents', 'nb_children', 'node_index', 'distance_to_end', 'nb_words_before', 'nb_nodes_per_depth'], help="The list of wanted features for the graph nodes.")
    parser.add_argument("-c", "--hidden-channels",type=int,default=64)
    parser.add_argument("-o", "--out-channels",type=int,default=2)
    parser.add_argument("-d", "--training-results-dir",type=str,default="training_results/",help="The directory where all the results and models produced during training are stored.")
    parser.add_argument("-s", "--metadata-hidden-size",type=int,default=15,help="The size taken (out of 64) by the metadata features in the hidden dimension.")
    parser.add_argument("-a", "--architecture", type=str, choices=["text","mixed","metadata"],default="text", help="Type of architecture and node features to use.")
    args = parser.parse_args()

    if not os.path.isdir(args.training_results_dir):
        if args.verbose:
            print(f"Directory {args.training_results_dir} does not exist. Creating it.")
        os.mkdir(args.training_results_dir)
    model_dir = os.path.join(args.training_results_dir,"model")
    explanation_subgraphs_dir = os.path.join(args.training_results_dir,"explanation_subgraphs")
    feature_importance_dir = os.path.join(args.training_results_dir,"feature_importance")
    if not os.path.isdir(model_dir):
        print(f"Directory {model_dir} does not exist. Creating it.")
        os.mkdir(model_dir)
    if not os.path.isdir(explanation_subgraphs_dir):
        print(f"Directory {explanation_subgraphs_dir} does not exist. Creating it.")
        os.mkdir(explanation_subgraphs_dir)
    if not os.path.isdir(feature_importance_dir):
        print(f"Directory {feature_importance_dir} does not exist. Creating it.")
        os.mkdir(feature_importance_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model_path = os.path.join(model_dir,"model.pth")

    if args.architecture != "text" or "train" in args.actions:
        train_graphs_with_full_features, eval_graphs_with_full_features = load_train_eval_graphs_with_full_features(args.graphs_dir)
        train_graphs, train_features, train_labels, train_embeddings = zip(*train_graphs_with_full_features)
        eval_graphs, eval_features, eval_labels, eval_embeddings = zip(*eval_graphs_with_full_features)
        print(f"type(train_features): {type(train_features)}, {type(train_features[0])}")

        if args.architecture != "text":
            if args.verbose:
                print(f"Normalizing metadata features.")
            if len(args.wanted_features) < 6:
                train_features = restrict_features(train_features, args.wanted_features)
                eval_features = restrict_features(eval_features, args.wanted_features)
            wanted_features = {feature:i for i, feature in enumerate(args.wanted_features)}
            train_features = normalize_node_index_per_graph(list(train_features), wanted_features)
            eval_features = normalize_node_index_per_graph(list(eval_features), wanted_features)
            train_features, eval_features, features_mean, features_std = normalize_dataset_features(train_features, eval_features, wanted_features)

    if "train" in args.actions:
        if args.architecture == "text":
            train_features = train_embeddings
            eval_features = eval_embeddings
        elif args.architecture == "mixed":
            train_features = [torch.cat((torch.tensor(e).to(device),torch.tensor(f).to(device)),dim=1) for e,f in zip(train_embeddings, train_features)]
            eval_features = [torch.cat((torch.tensor(e).to(device),torch.tensor(f).to(device)),dim=1) for e,f in zip(eval_embeddings, eval_features)]

        if args.verbose:
            print("Building the dataloader.")
        train_loader = build_dataloader(train_features, train_graphs, train_labels, batch_size=args.batch_size)
        eval_loader = build_dataloader(eval_features, eval_graphs, eval_labels, batch_size=args.batch_size)

        if args.architecture == "mixed":
            in_channels = (train_embeddings[0].shape[1],len(args.wanted_features))
        else:
            in_channels = train_features[0].shape[1]

        

        trained_model = train(
            train_dataloader=train_loader,
            val_loader=eval_loader,
            trained_model_path=trained_model_path, 
            in_channels=in_channels,
            out_channels=args.out_channels,
            hidden=args.hidden_channels,
            model_dir=model_dir,
            architecture=args.architecture,
            metadata_size=args.metadata_hidden_size,
            epochs=args.epochs,
            lr=args.learning_rate,
            decay=args.weight_decay
        )

    if "test" in args.actions:
        if args.verbose:
            print("Loading the test graphs and features.")
        test_graphs_with_full_features = {}
        for file in os.listdir(args.graphs_dir):
            dataset, lrm, label = file.split('.')[0].split('_')[1:4]
            if (dataset,lrm,label) not in test_graphs_with_full_features:
                test_graphs_with_full_features[(dataset,lrm,label)] = []
            with open(os.path.join(args.graphs_dir,file),"r") as f:
                lines = f.readlines()
                if "test" in file:
                    for line in lines:
                        sample = json.loads(line, object_hook=customJsonDecoder)
                        test_graphs_with_full_features[(dataset,lrm,label)].append((nx.from_dict_of_dicts(sample["graph"]),torch.tensor(sample["features"]),sample["label"], torch.tensor(sample["embeddings"])))
        k = list(test_graphs_with_full_features.keys())[0]
        if args.architecture == "mixed":
            in_channels = (test_graphs_with_full_features[k][0][3].shape[1],test_graphs_with_full_features[k][0][1].shape[1])
            trained_model = MixedGAT(in_channels[0],in_channels[1],args.out_channels,device,args.hidden_channels,args.metadata_hidden_size)
        elif args.architecture == "text":
            in_channels = test_graphs_with_full_features[k][0][3].shape[1]
            trained_model = TextGAT(in_channels, args.out_channels, device, args.hidden_channels)
        else:
            in_channels = test_graphs_with_full_features[k][0][1].shape[1]
            trained_model = GAT(in_channels, args.out_channels, args.hidden_channels)
        trained_model.load_state_dict(torch.load(trained_model_path))

        accuracies = {}
        accuracies_by_dataset_lrm_label = {}
        test_wanted_features = {feature:i for i,feature in enumerate(args.wanted_features)}
        for (dataset,lrm,label) in test_graphs_with_full_features:
            test_graphs, test_features, test_labels, test_embeddings = zip(*test_graphs_with_full_features[(dataset,lrm,label)])
            
            if args.architecture != "text":
                if args.verbose:
                    print("Normalizing data features.")
                if len(args.wanted_features) < 6:
                    test_features = restrict_features(test_features, args.wanted_features)
                test_features = normalize_node_index_per_graph(test_features,test_wanted_features)
                test_features = normalize_test_features(test_features,test_wanted_features,features_mean, features_std)
            if args.architecture == "text":
                test_features = test_embeddings
            elif args.architecture == "mixed":
                test_features = [torch.cat((torch.tensor(e).to(device),torch.tensor(f).to(device)),dim=1) for e,f in zip(test_embeddings, test_features)]

            if args.verbose:
                print(f"Building test dataloader for {(dataset,lrm,label)}.")
            test_loader = build_dataloader(test_features,test_graphs,test_labels,args.batch_size)   

            if args.verbose:
                print(f"Testing for {(dataset,lrm,label)}")

            accuracy = test(test_loader,trained_model)
            accuracies[(dataset,lrm,label)] = accuracy
            if dataset not in accuracies_by_dataset_lrm_label:
                accuracies_by_dataset_lrm_label[dataset] = []
            if lrm not in accuracies_by_dataset_lrm_label:
                accuracies_by_dataset_lrm_label[lrm] = []
            if label not in accuracies_by_dataset_lrm_label:
                accuracies_by_dataset_lrm_label[label] = []
            if (dataset,lrm) not in accuracies_by_dataset_lrm_label:
                accuracies_by_dataset_lrm_label[(dataset,lrm)] = []
            if (dataset,label) not in accuracies_by_dataset_lrm_label:
                accuracies_by_dataset_lrm_label[(dataset,label)] = []
            if (lrm,label) not in accuracies_by_dataset_lrm_label:
                accuracies_by_dataset_lrm_label[(lrm,label)] = []
            accuracies_by_dataset_lrm_label[label].append(accuracy)
            accuracies_by_dataset_lrm_label[dataset].append(accuracy)
            accuracies_by_dataset_lrm_label[lrm].append(accuracy)
            accuracies_by_dataset_lrm_label[(dataset,label)].append(accuracy)
            accuracies_by_dataset_lrm_label[(dataset,lrm)].append(accuracy)
            accuracies_by_dataset_lrm_label[(lrm,label)].append(accuracy)

        for el in accuracies_by_dataset_lrm_label:
            print(f"The accuracy for {el} is {sum(accuracies_by_dataset_lrm_label[el])/len(accuracies_by_dataset_lrm_label[el])}")

            