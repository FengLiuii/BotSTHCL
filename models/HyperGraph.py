import torch
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from torch_geometric.nn import HypergraphConv  # 假设有一个超图卷积层
from torch_geometric.data import Data
from tqdm import tqdm
import community as community_louvain
import networkx as nx
from networkx.algorithms.community import girvan_newman,asyn_fluidc
import torch.nn.functional as F
from torch_geometric.utils import to_networkx
import random
from torch_geometric.utils import k_hop_subgraph, subgraph, degree
from torch_sparse import SparseTensor
import networkx as nx
import community as community_louvain  # 确保已安装: pip install python-louvain
import numpy as np
import copy
from collections import defaultdict



class HyperGCL_Louvain_P(nn.Module):   
    def __init__(self, hidden_dim, n_heads, dropout):
        super(HyperGCL_Louvain_P, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.activation = nn.PReLU()

         # Define two layers of hypergraph convolution
        self.hypergraph_conv1 = HypergraphConv(hidden_dim, hidden_dim)
        self.hypergraph_conv2 = HypergraphConv(hidden_dim, hidden_dim)


        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.contrastive_loss_weight = nn.Parameter(torch.tensor(1.0))  
        self.init_weights()

    def forward(self, x, edge_index):
        data = Data(x=x, edge_index=edge_index)
        hyper_edge_index = self.build_hypergraph_from_graph(data)
        hyper_edge_index = hyper_edge_index.to(x.device)

        h1 = self.hypergraph_conv1(x, hyper_edge_index)
        h1 = self.activation(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)  # Apply dropout

        # Apply the second hypergraph convolution layer
        h2 = self.hypergraph_conv2(h1, hyper_edge_index)
        h2 = self.activation(h2)

        h_pos = self.projector(h1+x) 
        
        h_neg = self.projector(self.negative_sampling(h_pos))

        contrastive_loss = self.compute_contrastive_loss(h_pos, h_neg)
        contrastive_loss_weight = torch.nn.functional.softplus(self.contrastive_loss_weight)  
        if self.training:  
            weighted_contrastive_loss = contrastive_loss * contrastive_loss_weight
        else:
            weighted_contrastive_loss = contrastive_loss * self.contrastive_loss_weight.detach()
        return h_pos, weighted_contrastive_loss

  
    

    def compute_contrastive_loss(self, h_pos, h_neg, temperature=0.1):
        """
        Compute InfoNCE loss between positive and negative samples.
        h_pos: Positive sample embeddings
        h_neg: Negative sample embeddings
        temperature: Temperature parameter for scaling
        """
        # Normalize embeddings
       

        h_pos = F.normalize(h_pos, dim=-1)
        h_neg = F.normalize(h_neg, dim=-1)
        
        # Cosine similarity
        positive_similarity = F.cosine_similarity(h_pos, h_neg, dim=-1) / temperature
        negative_similarity = F.cosine_similarity(h_pos.unsqueeze(1), h_neg, dim=-1) / temperature
        
        # Softmax-based contrastive loss
        loss = -torch.log(torch.exp(positive_similarity) / (torch.exp(positive_similarity) + torch.exp(negative_similarity).sum(dim=-1)))
        
        return loss.mean()



    def negative_sampling(self, x):
        
        return x[torch.randperm(x.size(0))]

    def build_hypergraph_from_graph(self, data):
        G = nx.Graph()
        edge_list = data.edge_index.t().cpu().numpy().tolist()
        if len(edge_list) == 0:
            num_nodes = data.x.shape[0]
            if num_nodes > 1:
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_list.append([i, j])
            elif num_nodes == 1:
                edge_list.append([0, 0])

        G.add_edges_from(edge_list)
        partition = community_louvain.best_partition(G)
        if not partition:
            partition = {node: 0 for node in G.nodes()}

        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        edge_list = []
        for community_nodes in communities.values():
            if len(community_nodes) > 1:
                for i in range(len(community_nodes)):
                    for j in range(i + 1, len(community_nodes)):
                        edge_list.append([community_nodes[i], community_nodes[j]])
            elif len(community_nodes) == 1:
                edge_list.append([community_nodes[0], community_nodes[0]])

        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()
        return hyperedges

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()




            

def create_hypersubgraph(data, args):
    sub_size = args.sub_size
    node_size = int(data.n_x[0].item())
    hyperedge_size = int(data.num_hyperedges)
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        sample_nodes, 1, edge_index, relabel_nodes=False, flow="target_to_source"
    )
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    # relabel
    node_idx = torch.zeros(
        2 * node_size + hyperedge_size, dtype=torch.long, device=device
    )
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    x = data.x[sample_nodes]
    data_sub = Data(x=x, edge_index=sub_edge_index)
    data_sub.n_x = torch.tensor([sub_size], device=device)
    data_sub.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * sub_size], device=device)
    data_sub.norm = 0
    data_sub.totedges = torch.tensor([sub_nodes.size(0) - sub_size], device=device)
    data_sub.num_ori_edge = sub_edge_index.shape[1] - sub_size
    return data_sub, sorted(set([i for i in range(data.x.shape[0])])), sorted(
        set(
            edge_index[1][
                torch.where(
                    (edge_index[1] < node_size + data.num_hyperedges)
                    & (edge_index[1] > node_size - 1)
                )[0]
            ]
            .cpu()
            .numpy()
        )
    )

def permute_edges(data, aug_ratio, permute_self_edge, args):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    permute_num = int((edge_num - node_num) * aug_ratio)
    edge_index_orig = copy.deepcopy(data.edge_index)
    edge_index = data.edge_index.cpu().numpy()

    if args.add_e:
        idx_add_1 = np.random.choice(node_num, permute_num)
        idx_add_2 = np.random.choice(int(data.num_hyperedges), permute_num)
        idx_add = np.stack((idx_add_1, idx_add_2), axis=0)
    edge2remove_index = np.where(edge_index[1] < data.num_hyperedges.item())[0]
    edge2keep_index = np.where(edge_index[1] >= data.num_hyperedges.item())[0]

    try:
        edge_keep_index = np.random.choice(
            edge2remove_index, (edge_num - node_num) - permute_num, replace=False
        )
    except ValueError:
        edge_keep_index = np.random.choice(
            edge2remove_index, (edge_num - node_num) - permute_num, replace=True
        )

    edge_after_remove1 = edge_index[:, edge_keep_index]
    edge_after_remove2 = edge_index[:, edge2keep_index]
    if args.add_e:
        edge_index = np.concatenate(
            (
                edge_after_remove1,
                edge_after_remove2,
            ),
            axis=1,
        )
    else:
        edge_index = np.concatenate((edge_after_remove1, edge_after_remove2), axis=1)
    data.edge_index = torch.tensor(edge_index, device=data.edge_index.device)
    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                edge_index_orig[1][
                    torch.where(
                        (edge_index_orig[1] < node_num + data.num_hyperedges)
                        & (edge_index_orig[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )

def permute_hyperedges(data, aug_ratio):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = int(data.num_hyperedges)
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index_orig = copy.deepcopy(data.edge_index)
    edge_index = data.edge_index.cpu().numpy()
    edge_remove_index = np.random.choice(hyperedge_num, permute_num, replace=False)
    edge_remove_index_dict = {ind: i for i, ind in enumerate(edge_remove_index)}

    edge_remove_index_all = [
        i for i, he in enumerate(edge_index[1]) if he in edge_remove_index_dict
    ]
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove

    data.edge_index = torch.tensor(edge_index, device=data.edge_index.device)

    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                edge_index_orig[1][
                    torch.where(
                        (edge_index_orig[1] < node_num + data.num_hyperedges)
                        & (edge_index_orig[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )

def adapt(data, aug_ratio, aug):
    node_num, _ = data.x.size()
    _, edge_num = data.edge_index.size()
    hyperedge_num = int(data.num_hyperedges)
    permute_num = int(hyperedge_num * aug_ratio)
    index = defaultdict(list)
    edge_index = data.edge_index.cpu().numpy()
    for i, he in enumerate(edge_index[1]):
        index[he].append(i)
    # edge
    edge_index_orig = copy.deepcopy(data.edge_index)
    drop_weights = degree_drop_weights(data.edge_index, hyperedge_num)
    edge_index_1 = drop_edge_weighted(
        data.edge_index,
        drop_weights,
        p=aug_ratio,
        threshold=0.7,
        h=hyperedge_num,
        index=index,
    )

    # feature
    edge_index_ = data.edge_index
    node_deg = degree(edge_index_[0], num_nodes=data.x.size(0))
    feature_weights = feature_drop_weights(data.x, node_c=node_deg)
    x_1 = drop_feature_weighted(data.x, feature_weights, aug_ratio, threshold=0.7)
    if aug == "adapt_edge":
        data.edge_index = edge_index_1
    elif aug == "adapt_feat":
        data.x = x_1
    else:
        data.edge_index = edge_index_1
        data.x = x_1
    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                edge_index_orig[1][
                    torch.where(
                        (edge_index_orig[1] < node_num + data.num_hyperedges)
                        & (edge_index_orig[1] > node_num - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )

def drop_feature_weighted(x, w, p: float, threshold: float = 0.7):
    w = w / w.mean() * p
    w = w.where(w < threshold, torch.ones_like(w) * threshold)
    drop_prob = w
    drop_mask = torch.bernoulli(drop_prob).to(torch.bool)
    x = x.clone()
    x[:, drop_mask] = 0.0
    return x

def degree_drop_weights(edge_index, h):
    edge_index_ = edge_index
    deg = degree(edge_index_[1], num_nodes=h)
    deg_col = deg
    s_col = torch.log(deg_col + 1e-9)
    weights = (s_col - s_col.min() + 1e-9) / (s_col.mean() - s_col.min() + 1e-9)
    return weights

def feature_drop_weights(x, node_c):
    x = torch.abs(x).to(torch.float32)
    # 100 x 2012 mat 2012-> 100
    w = x.t() @ node_c
    w = torch.log(w + 1e-7)
    s = (w - w.min()) / (w.mean() - w.min() + 1e-9)
    return s

def drop_edge_weighted(edge_index, edge_weights, p: float, h, index, threshold: float = 1.0):
    _, edge_num = edge_index.size()
    edge_weights = (edge_weights + 1e-9) / (edge_weights.mean() + 1e-9) * p
    edge_weights = edge_weights.where(
        edge_weights < threshold, torch.ones_like(edge_weights) * threshold
    )
    # 保留概率
    sel_mask = torch.bernoulli(edge_weights).to(torch.bool)
    edge_remove_index = np.array(list(range(h)))[sel_mask.cpu().numpy()]
    edge_remove_index_all = []
    for remove_index in edge_remove_index:
        edge_remove_index_all.extend(index[remove_index])
    edge_keep_index = list(set(list(range(edge_num))) - set(edge_remove_index_all))
    edge_after_remove = edge_index[:, edge_keep_index]
    edge_index = edge_after_remove
    return edge_index

def mask_nodes(data, aug_ratio):
    node_num, feat_dim = data.x.size()
    mask_num = int(node_num * aug_ratio)
    token = data.x.mean(dim=0)
    zero_v = torch.zeros_like(token)
    idx_mask = np.random.choice(node_num, mask_num, replace=False)
    data.x[idx_mask] = token
    return (
        data,
        sorted(set([i for i in range(data.x.shape[0])])),
        sorted(
            set(
                data.edge_index[1][
                    torch.where(
                        (data.edge_index[1] < data.x.size(0) + data.num_hyperedges)
                        & (data.edge_index[1] > data.x.size(0) - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )

def drop_nodes(data, aug_ratio):
    node_size = int(data.n_x[0].item())
    sub_size = int(node_size * (1 - aug_ratio))
    hyperedge_size = int(data.num_hyperedges)
    sample_nodes = np.random.permutation(node_size)[:sub_size]
    sample_nodes = list(np.sort(sample_nodes))
    edge_index = data.edge_index
    device = edge_index.device
    sub_nodes, sub_edge_index, mapping, _ = k_hop_subgraph(
        sample_nodes, 1, edge_index, relabel_nodes=False, flow="target_to_source"
    )
    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    sub_edge_index_orig = copy.deepcopy(sub_edge_index)
    # relabel
    node_idx = torch.zeros(
        2 * node_size + hyperedge_size, dtype=torch.long, device=device
    )
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    data.x = data.x[sample_nodes]
    data.edge_index = sub_edge_index

    data.n_x = torch.tensor([sub_size], device=device)
    data.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * sub_size], device=device)
    data.norm = 0
    data.totedges = torch.tensor([sub_nodes.size(0) - sub_size], device=device)
    data.num_ori_edge = sub_edge_index.shape[1] - sub_size

    return (
        data,
        sorted(set(sub_nodes[:sub_size].cpu().numpy())),
        sorted(
            set(
                sub_edge_index_orig[1][
                    torch.where(
                        (sub_edge_index_orig[1] < node_size + hyperedge_size)
                        & (sub_edge_index_orig[1] > node_size - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )

def subgraph_aug(data, aug_ratio, start):
    n_walkLen = 16
    node_num, _ = data.x.size()
    he_num = data.totedges.item()
    edge_index = data.edge_index

    device = edge_index.device

    row, col = edge_index
    adj = SparseTensor(
        row=torch.cat([row, col]),
        col=torch.cat([col, row]),
        sparse_sizes=(node_num + he_num, he_num + node_num),
    )

    node_idx = adj.random_walk(start.flatten(), n_walkLen).view(-1)
    sub_nodes = node_idx.unique()
    sub_nodes, _ = torch.sort(sub_nodes)

    node_size = int(data.n_x[0].item())
    hyperedge_size = int(data.num_hyperedges)
    sub_edge_index, _, hyperedge_idx = subgraph(
        sub_nodes, edge_index, relabel_nodes=False, return_edge_mask=True
    )

    sub_nodes, sorted_idx = torch.sort(sub_nodes)
    sub_edge_index_orig = copy.deepcopy(sub_edge_index)
    # relabel
    node_idx = torch.zeros(
        2 * node_size + hyperedge_size, dtype=torch.long, device=device
    )
    node_idx[sub_nodes] = torch.arange(sub_nodes.size(0), device=device)
    sub_edge_index = node_idx[sub_edge_index]
    node_keep_idx = sub_nodes[torch.where(sub_nodes < node_size)[0]]
    data.x = data.x[node_keep_idx]
    data.edge_index = sub_edge_index

    data.n_x = torch.tensor([node_keep_idx.size(0)], device=device)
    data.num_hyperedges = torch.tensor([sub_nodes.size(0) - 2 * node_keep_idx.size(0)], device=device)
    data.norm = 0
    data.totedges = torch.tensor([sub_nodes.size(0) - node_keep_idx.size(0)], device=device)
    data.num_ori_edge = sub_edge_index.shape[1] - node_keep_idx.size(0)
    return (
        data,
        sorted(set(node_keep_idx.cpu().numpy().tolist())),
        sorted(
            set(
                sub_edge_index_orig[1][
                    torch.where(
                        (sub_edge_index_orig[1] < node_size + hyperedge_size)
                        & (sub_edge_index_orig[1] > node_size - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        ),
    )

def aug(data, aug_type, args, start=None):
    data_aug = copy.deepcopy(data)
    if aug_type == "mask":
        data_aug, sample_nodes, sample_hyperedge = mask_nodes(data_aug, args.aug_ratio)
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "edge":
        data_aug, sample_nodes, sample_hyperedge = permute_edges(
            data_aug, args.aug_ratio, args.permute_self_edge, args
        )
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "hyperedge":
        data_aug, sample_nodes, sample_hyperedge = permute_hyperedges(data_aug, args.aug_ratio)
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "subgraph":
        data_aug, sample_nodes, sample_hyperedge = subgraph_aug(
            data_aug, args.aug_ratio, start
        )
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "drop":
        data_aug, sample_nodes, sample_hyperedge = drop_nodes(data_aug, args.aug_ratio)
        return data_aug, sample_nodes, sample_hyperedge
    elif aug_type == "none":
        return data_aug, sorted(set([i for i in range(data_aug.x.shape[0])])), sorted(
            set(
                data_aug.edge_index[1][
                    torch.where(
                        (data_aug.edge_index[1] < data_aug.x.size(0) + data_aug.num_hyperedges)
                        & (data_aug.edge_index[1] > data_aug.x.size(0) - 1)
                    )[0]
                ]
                .cpu()
                .numpy()
            )
        )
    elif "adapt" in aug_type:
        data_aug, sample_nodes, sample_hyperedge = adapt(data_aug, args.aug_ratio, aug_type)
        return data_aug, sample_nodes, sample_hyperedge
    else:
        raise ValueError(f"Unsupported augmentation type: {aug_type}")
    return data_aug, sample_nodes, sample_hyperedge

class HyperGCL_Louvain_P_DA(nn.Module):
    def __init__(self, hidden_dim, n_heads, dropout, args=None):
        super(HyperGCL_Louvain_P_DA, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.dropout = dropout
        self.args = args  # Augmentation configuration parameters
        
        # Define two layers of hypergraph convolution
        self.hypergraph_conv = HypergraphConv(hidden_dim, hidden_dim)
        

        self.activation = nn.PReLU()

        # Projector for contrastive learning
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.contrastive_loss_weight = nn.Parameter(torch.tensor(1.0))  # Contrastive loss weight initialization
        self.init_weights()

    def forward(self, x, edge_index, start=None):
        # Apply data augmentation during training
        if self.training and self.args is not None:
            x_aug, edge_index_aug = self.apply_augmentations(x, edge_index, start)
        else:
            x_aug, edge_index_aug = x, edge_index

        data = Data(x=x_aug, edge_index=edge_index_aug)
        data.n_x = torch.tensor([x_aug.size(0)], device=x.device)
        data.num_hyperedges = torch.tensor([0], device=x.device)  # Adjust based on your needs
        data.totedges = torch.tensor([0], device=x.device)        # Adjust based on your needs
        data.num_ori_edge = torch.tensor([edge_index_aug.size(1)], device=x.device)
        # print(data,data.n_x,data.num_hyperedges,data.totedges,data.num_ori_edge)
        # Build hypergraph and obtain hypergraph edge index
        hyper_edge_index = self.build_hypergraph_from_graph(data)
        hyper_edge_index = hyper_edge_index.to(x.device)

        # Apply two layers of Hypergraph Convolution
        h = self.hypergraph_conv(x_aug, hyper_edge_index)
        h = self.activation(h)
        h = F.dropout(h, p=self.dropout, training=self.training)  # Apply dropout

        

        # Project positive samples using projector
        h_pos = self.projector(h+x)

        # Generate negative samples through negative sampling
        h_neg = self.projector(self.negative_sampling(h_pos))

        # Compute contrastive loss
        contrastive_loss = self.compute_contrastive_loss(h_pos, h_neg)
        contrastive_loss_weight = F.softplus(self.contrastive_loss_weight)  # Ensure the weight is positive

        if self.training:  # Update contrastive loss weight only during training
            weighted_contrastive_loss = contrastive_loss * contrastive_loss_weight
        else:
            weighted_contrastive_loss = contrastive_loss * self.contrastive_loss_weight.detach()

        return h_pos, weighted_contrastive_loss

    def compute_contrastive_loss(self, h_pos, h_neg, temperature=0.1):
        """
        Compute InfoNCE loss between positive and negative samples.
        h_pos: Positive sample embeddings
        h_neg: Negative sample embeddings
        temperature: Temperature parameter for scaling
        """
        # Normalize embeddings
       

        h_pos = F.normalize(h_pos, dim=-1)
        h_neg = F.normalize(h_neg, dim=-1)
        
        # Cosine similarity
        positive_similarity = F.cosine_similarity(h_pos, h_neg, dim=-1) / temperature
        negative_similarity = F.cosine_similarity(h_pos.unsqueeze(1), h_neg, dim=-1) / temperature
        
        # Softmax-based contrastive loss
        loss = -torch.log(torch.exp(positive_similarity) / (torch.exp(positive_similarity) + torch.exp(negative_similarity).sum(dim=-1)))
        
        return loss.mean()

    def negative_sampling(self, x):
        # Simple negative sampling by shuffling node features
        return x[torch.randperm(x.size(0))]

    def build_hypergraph_from_graph(self, data):
        G = nx.Graph()
        edge_list = data.edge_index.t().cpu().numpy().tolist()
        if len(edge_list) == 0:
            num_nodes = data.x.shape[0]
            if num_nodes > 1:
                for i in range(num_nodes):
                    for j in range(i + 1, num_nodes):
                        edge_list.append([i, j])
            elif num_nodes == 1:
                edge_list.append([0, 0])

        G.add_edges_from(edge_list)
        partition = community_louvain.best_partition(G)
        if not partition:
            partition = {node: 0 for node in G.nodes()}

        communities = {}
        for node, community in partition.items():
            if community not in communities:
                communities[community] = []
            communities[community].append(node)

        edge_list = []
        for community_nodes in communities.values():
            if len(community_nodes) > 1:
                for i in range(len(community_nodes)):
                    for j in range(i + 1, len(community_nodes)):
                        edge_list.append([community_nodes[i], community_nodes[j]])
            elif len(community_nodes) == 1:
                edge_list.append([community_nodes[0], community_nodes[0]])

        hyperedges = torch.tensor(edge_list, dtype=torch.long).t().contiguous().clone().detach()
        return hyperedges

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()

    def apply_augmentations(self, x, edge_index, start=None):
        """
        Apply the specified data augmentations based on the configuration.
        """
        if self.args is None:
            raise ValueError("args must be provided for augmentation.")

        # Create a Data object
        data = Data(x=x, edge_index=edge_index)
        data.n_x = torch.tensor([x.size(0)], device=x.device)
        data.num_hyperedges = torch.tensor([0], device=x.device)  # Adjust based on your needs
        data.totedges = torch.tensor([0], device=x.device)        # Adjust based on your needs
        data.num_ori_edge = torch.tensor([edge_index.size(1)], device=x.device)

        # Choose augmentation type and parameters
        aug_type = self.args.aug_type         # Augmentation type e.g., 'mask', 'edge', 'hyperedge', etc.
        aug_ratio = self.args.aug_ratio       # Augmentation ratio
        permute_self_edge = getattr(self.args, 'permute_self_edge', False)  # Whether to permute self edges
        add_e = getattr(self.args, 'add_e', False)  # Whether to add new edges (used only in 'edge' augmentation)

        # Call the corresponding augmentation function
        if aug_type == "mask":
            data_aug, sample_nodes, sample_hyperedge = mask_nodes(data, aug_ratio)
        elif aug_type == "edge":
            data_aug, sample_nodes, sample_hyperedge = permute_edges(
                data, aug_ratio, permute_self_edge, self.args
            )
        elif aug_type == "hyperedge":
            data_aug, sample_nodes, sample_hyperedge = permute_hyperedges(data, aug_ratio)
        elif aug_type == "subgraph":
            data_aug, sample_nodes, sample_hyperedge = subgraph_aug(
                data, aug_ratio, start
            )
        elif aug_type == "drop":
            data_aug, sample_nodes, sample_hyperedge = drop_nodes(data, aug_ratio)
        elif aug_type == "none":
            data_aug = data  # No augmentation
            sample_nodes = sorted(set([i for i in range(data_aug.x.shape(0))]))
            sample_hyperedge = sorted(
                set(
                    data_aug.edge_index[1][
                        torch.where(
                            (data_aug.edge_index[1] < data_aug.x.size(0) + data_aug.num_hyperedges)
                            & (data_aug.edge_index[1] > data_aug.x.size(0) - 1)
                        )[0]
                    ]
                    .cpu()
                    .numpy()
                )
            )
        elif "adapt" in aug_type:
            data_aug, sample_nodes, sample_hyperedge = adapt(data, aug_ratio, aug_type)
        else:
            raise ValueError(f"Unsupported augmentation type: {aug_type}")

        # Return the augmented features and edge index
        return data_aug.x, data_aug.edge_index