import torch
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.nn import global_mean_pool, global_max_pool

class GAT(torch.nn.Module):
    def __init__(self, in_channels:int, out_channels:int, hidden:int=64, num_heads=4):
        super().__init__()
        self.hidden = hidden
        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=self.hidden, heads=num_heads, edge_dim=3)
        self.conv2 = GATv2Conv(in_channels=self.hidden*num_heads, out_channels=self.hidden, edge_dim=3, concat=False)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.BatchNorm1d(self.hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(self.hidden, out_channels)
        )

    def forward(self, x, edge_index, edge_attr, batch):
       
        x = self.conv1(x, edge_index, edge_attr=edge_attr)
        x= torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = global_mean_pool(x, batch)
        x = self.linear(x)

        return torch.nn.functional.log_softmax(x, dim=1)

class TextGAT(torch.nn.Module):
    def __init__(self, in_channels, out_channels, device:str, hidden:int=64, num_heads=4):
        super().__init__()
        self.device=device
        self.hidden=hidden
        self.conv1 = GATv2Conv(in_channels=in_channels, out_channels=hidden, heads=num_heads, edge_dim=3)
        self.conv2 = GATv2Conv(in_channels=self.hidden*num_heads, out_channels=hidden, edge_dim=3, concat=False)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(self.hidden, self.hidden),
            torch.nn.BatchNorm1d(self.hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(self.hidden, out_channels)
        )

    def forward(self, x, edge_index, text_attr, batch):
        x = x.to(self.device)
        x = self.conv1(x, edge_index, edge_attr=text_attr)
        x= torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index, edge_attr=text_attr)
        x = global_max_pool(x, batch)
        x = self.linear(x)

        return torch.nn.functional.log_softmax(x, dim=1)
    
class MixedGAT(torch.nn.Module):
    def __init__(self, text_dim:int, features_dim:int, out_channels:int, device:str, hidden:int, metadata_size:int, num_heads:int=1):
        super().__init__()
        self.device = device
        self.text_dim = text_dim
        self.linear_text = torch.nn.Linear(text_dim, hidden-metadata_size)
        self.linear_metadata = torch.nn.Linear(features_dim, metadata_size)
        self.conv1 = GATv2Conv(in_channels=hidden, out_channels=hidden, edge_dim=3, heads=num_heads, concat=False)
        self.conv2 = GATv2Conv(in_channels=hidden*num_heads, out_channels=hidden, edge_dim=3, concat=False)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            torch.nn.BatchNorm1d(hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.4),
            torch.nn.Linear(hidden, out_channels)
        )

    def forward(self, x, edge_index, text_attr, batch):
        x_text = x[:,:self.text_dim]
        x_metadata = x[:,self.text_dim:]
        x_text = x_text.to(self.device)
        x_text = self.linear_text(x_text)
        print(x_text.size())
        print(x_metadata.size())
        x_metadata = x_metadata.to(self.device)
        x_metadata = self.linear_metadata(x_metadata)
        x = torch.cat((x_text, x_metadata), dim=1)
        x = self.conv1(x, edge_index, edge_attr=text_attr)
        x= torch.nn.functional.relu(x)
        x = self.conv2(x, edge_index, edge_attr=text_attr)
        x = global_max_pool(x, batch)
        x = self.linear(x)

        return torch.nn.functional.log_softmax(x, dim=1)