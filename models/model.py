# # -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
from models.GraphStructuralLayer import GraphStructuralLayer
from models.GraphTemporalLayer import GraphTemporalLayer
from models.NodeFeatureEmbeddingLayer import NodeFeatureEmbeddingLayer
from models.PositionEmbeddingLayer import PositionEncodingClusteringCoefficient, PositionEncodingBidirectionalLinks
from models.HyperGraph import   HyperGCL_AsynFluidc,HyperGCL_Louvain_P, HyperGCL_Louvain_P_DA
from tqdm import tqdm






class BotSTHCL(nn.Module):
    def __init__(self, args):
        super(BotSTHCL, self).__init__()
        self.args = args
        self.hidden_dim = args.hidden_dim
        self.structural_head_config = args.structural_head_config
        self.structural_drop = args.structural_drop
        self.temporal_head_config = args.temporal_head_config
        self.temporal_drop = args.temporal_drop
        self.window_size = args.window_size
        self.temporal_module_type = args.temporal_module_type

       
        self.contrastive_loss_weight = nn.Parameter(torch.tensor(1.0))  

        
        self.node_feature_embedding_layer, self.position_encoding_clustering_coefficient_layer, \
        self.position_encoding_bidirectional_links_ratio_layer, self.structural_layer, \
        self.temporal_layer = self.build_model()

    def forward(self, all_snapshots_des_tensor, all_snapshots_tweet_tensor, all_snapshots_num_prop,
                all_snapshots_category_prop, all_snapshots_edge_index, all_snapshots_clustering_coefficient,
                all_snapshots_bidirectional_links_ratio, all_snapshots_exist_nodes, current_batch_size):
        all_snapshots_structural_output = []
        total_contrastive_loss = 0.0  
        num_of_snapshot = len(all_snapshots_des_tensor)  

        
        for t in range(num_of_snapshot):
            one_snapshot_des_tensor = all_snapshots_des_tensor[t]
            one_snapshot_tweet_tensor = all_snapshots_tweet_tensor[t]
            one_snapshot_num_prop = all_snapshots_num_prop[t]
            one_snapshot_category_prop = all_snapshots_category_prop[t]

           
            x = self.node_feature_embedding_layer(one_snapshot_des_tensor, one_snapshot_tweet_tensor, 
                                                  one_snapshot_num_prop, one_snapshot_category_prop)
            one_snapshot_edge_index = all_snapshots_edge_index[t]  

            
            output, contrastive_loss = self.structural_layer(x, one_snapshot_edge_index)
            total_contrastive_loss += contrastive_loss  
            all_snapshots_structural_output.append(output[:current_batch_size])

        
        all_snapshots_structural_output = torch.stack(all_snapshots_structural_output, dim=1)  # [batch_size, num_snapshots, hidden_dim]

        if torch.any(torch.isnan(all_snapshots_structural_output)):  
            print('structural_output has nan')

        
        position_embedding_clustering_coefficient = [
            self.position_encoding_clustering_coefficient_layer(all_snapshots_clustering_coefficient[t])[:current_batch_size]
            for t in range(num_of_snapshot)
        ]
        position_embedding_bidirectional_links_ratio = [
            self.position_encoding_bidirectional_links_ratio_layer(all_snapshots_bidirectional_links_ratio[t])[:current_batch_size]
            for t in range(num_of_snapshot)
        ]

        
        position_embedding_clustering_coefficient = torch.stack(position_embedding_clustering_coefficient, dim=1)
        position_embedding_bidirectional_links_ratio = torch.stack(position_embedding_bidirectional_links_ratio, dim=1)

       
        exist_nodes = all_snapshots_exist_nodes.transpose(0, 1)

       
        temporal_output = self.temporal_layer(all_snapshots_structural_output, 
                                              position_embedding_clustering_coefficient,
                                              position_embedding_bidirectional_links_ratio, exist_nodes)

        if torch.any(torch.isnan(temporal_output)):  
            print('temporal_output has nan')

        
        contrastive_loss_weight = torch.nn.functional.softplus(self.contrastive_loss_weight)

       
        return temporal_output, total_contrastive_loss * contrastive_loss_weight

    def build_model(self):
        
        node_feature_embedding_layer = NodeFeatureEmbeddingLayer(dataset_name=self.args.dataset_name,hidden_dim=self.hidden_dim)
        
        
        position_encoding_clustering_coefficient_layer = PositionEncodingClusteringCoefficient(hidden_dim=self.hidden_dim)
        
       
        position_encoding_bidirectional_links_ratio_layer = PositionEncodingBidirectionalLinks(hidden_dim=self.hidden_dim)
        
     
        structural_layer = HyperGCL_Louvain_P_DA(hidden_dim=self.hidden_dim, 
                                            n_heads=self.structural_head_config, 
                                            dropout=self.structural_drop)
        
        
        temporal_layer = GraphTemporalLayer(hidden_dim=self.hidden_dim,
                                            n_heads=self.temporal_head_config,
                                            dropout=self.temporal_drop,
                                            num_time_steps=self.window_size,
                                            temporal_module_type=self.temporal_module_type)
        
        
        return node_feature_embedding_layer, position_encoding_clustering_coefficient_layer, \
               position_encoding_bidirectional_links_ratio_layer, structural_layer, temporal_layer
