from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from itertools import accumulate
from models.coarse.model_components import BertAttention, LinearLayer, \
                                            TrainablePositionalEncoding, CMMT_SC
from models.coarse.language_encoder import LanguageEncoder
from models.coarse.object_encoder import ObjectEncoder

from easydict import EasyDict as edict


def mask_logits(target, mask):
        return target * mask + (1 - mask) * (-1e10)


class CellRetrievalNetwork(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], args
    ):
        """Module for global place recognition.
        Implemented as a text branch (language encoder) and a 3D submap branch (object encoder).
        The 3D submap branch aggregates information about a varying count of multiple objects through Attention.
        """
        super(CellRetrievalNetwork, self).__init__()
        self.embed_dim = args.coarse_embed_dim

        """
        3D submap branch
        """

        # CARE: possibly handle variation in forward()!

        self.object_encoder = ObjectEncoder(args.coarse_embed_dim, known_classes, known_colors, args)
        self.object_size = args.object_size
        self.object_pos_embed = TrainablePositionalEncoding(max_position_embeddings=2000,
                                                          hidden_size=args.coarse_embed_dim,dropout=args.input_drop)
        self.cell_input_proj = LinearLayer(args.coarse_embed_dim, args.coarse_embed_dim, layer_norm=True,
                                            dropout=args.input_drop, relu=True)    
        self.cell_encoder1 = CMMT_SC(edict(hidden_size=args.coarse_embed_dim, intermediate_size=args.coarse_embed_dim,
                                                 hidden_dropout_prob=args.drop, num_attention_heads=args.n_heads,
                                                 attention_probs_dropout_prob=args.drop,object_size=args.object_size,sft_factor=args.sft_factor))        
        self.weight_token = nn.Parameter(torch.randn(1, 1, args.coarse_embed_dim))

        """
        Textual branch
        """
        self.language_encoder = LanguageEncoder(args.coarse_embed_dim,  
                                                hungging_model = args.hungging_model, 
                                                fixed_embedding = args.fixed_embedding, 
                                                intra_module_num_layers = args.intra_module_num_layers, 
                                                intra_module_num_heads = args.intra_module_num_heads, 
                                                is_fine = False,  
                                                inter_module_num_layers = args.inter_module_num_layers,
                                                inter_module_num_heads = args.inter_module_num_heads,
                                                ) 


        print(
            f"CellRetrievalNetwork, class embed {args.class_embed}, color embed {args.color_embed}, dim: {args.coarse_embed_dim}, features: {args.use_features}"
        )

    
    @staticmethod
    def encode_input(feat, mask, input_proj_layer, encoder_layer, pos_embed_layer,weight_token=None):
        """
        Args:
            feat: (N, L, D_input), torch.float32
            mask: (N, L), torch.float32, with 1 indicates valid query, 0 indicates mask
            input_proj_layer: down project input
            encoder_layer: encoder layer
            pos_embed_layer: positional embedding layer
        """

        feat = input_proj_layer(feat)
        feat = pos_embed_layer(feat)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (N, 1, L), torch.FloatTensor
        if weight_token is not None:
 
            return encoder_layer(feat, mask, weight_token)

        else:
            return encoder_layer(feat, mask)  # (N, L, D_hidden)
    
    
    
    def encode_text(self, descriptions):

        description_encodings = self.language_encoder(descriptions)  # [B, DIM]

        description_encodings = F.normalize(description_encodings)

        return description_encodings


    def encode_objects(self, objects, object_points):
        """
        Process the objects in a flattened way to allow for the processing of batches with uneven sample counts
        """
        
        batch = []  # Batch tensor to send into PyG


        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:

                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)
        embeddings, pos_postions= self.object_encoder(objects, object_points)
        object_size = self.object_size

        index_list = [0]
        last = 0
        
        x = torch.zeros(len(objects), object_size, self.embed_dim).to(self.device)
      

        for obj in objects:
            index_list.append(last + len(obj))
            last += len(obj)
        
        embeddings = F.normalize(embeddings, dim=-1)  

        for idx in range(len(index_list) - 1):
            num_object_raw = index_list[idx + 1] - index_list[idx]
            start = index_list[idx]
            num_object = num_object_raw if num_object_raw <= object_size else object_size
            x[idx,: num_object] = embeddings[start : (start + num_object)]
            
        mask = np.ones((len(objects),object_size), np.int_)
        mask = torch.from_numpy(mask)
        mask = mask.to(x.device)

        x = self.encode_input(x,mask,self.cell_input_proj,self.cell_encoder1,self.object_pos_embed,self.weight_token)
        x = torch.where(mask.unsqueeze(-1).repeat(1, 1, x.shape[-1]) == 1.0, \
                                                                        x, 0. * x)
        x = x.permute(1, 0, 2).contiguous()
        del embeddings, pos_postions
        
        x = x.max(dim = 0)[0]
        x = F.normalize(x)

        return x

    def forward(self):
        raise Exception("Not implemented.")

    @property
    def device(self):
        return self.language_encoder.device

    def get_device(self):
        return self.language_encoder.device

