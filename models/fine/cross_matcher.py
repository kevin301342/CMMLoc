from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import networkx as nx

from models.fine.language_encoder import get_mlp, LanguageEncoder
from models.fine.object_encoder import ObjectEncoder
from models.fine.transformer import TransformerDecoderLayer

from datapreparation.kitti360pose.imports import Object3d as Object3d_K360

def get_mlp_offset(dims: List[int], add_batchnorm=False) -> nn.Sequential:
    """Return an MLP without trailing ReLU or BatchNorm for Offset/Translation regression.

    Args:
        dims (List[int]): List of dimension sizes
        add_batchnorm (bool, optional): Whether to add a BatchNorm. Defaults to False.

    Returns:
        nn.Sequential: Result MLP
    """
    if len(dims) < 3:
        print("get_mlp(): less than 2 layers!")
    mlp = []
    for i in range(len(dims) - 1):
        mlp.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mlp.append(nn.ReLU())
            if add_batchnorm:
                mlp.append(nn.BatchNorm1d(dims[i + 1]))
    return nn.Sequential(*mlp)
        
def get_direction(pose0, pose):
    # closest_point = obj.get_closest_point(pose)
    obj2pose = pose0 - pose
    if np.linalg.norm(obj2pose[0:2]) < 0.05:  # Before: 0.015
        direction = "on-top"
    else:
        if abs(obj2pose[0]) >= abs(obj2pose[1]) and obj2pose[0] >= 0:
            direction = "east"
        if abs(obj2pose[0]) >= abs(obj2pose[1]) and obj2pose[0] <= 0:
            direction = "west"
        if abs(obj2pose[0]) <= abs(obj2pose[1]) and obj2pose[1] >= 0:
            direction = "north"
        if abs(obj2pose[0]) <= abs(obj2pose[1]) and obj2pose[1] <= 0:
            direction = "south"
    return direction

class CrossMatch(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], args
    ):
        """Fine localization module.
        Consists of text branch (language encoder) and a 3D submap branch (object encoder) and
        cascaded cross-attention transformer (CCAT) module.

        Args:
            known_classes (List[str]): List of known classes
            known_colors (List[str]): List of known colors
            args: Global training args
        """
        super(CrossMatch, self).__init__()
        self.embed_dim = args.fine_embed_dim

        self.object_encoder = ObjectEncoder(args.fine_embed_dim, known_classes, known_colors, args)

        self.language_encoder = LanguageEncoder(args.fine_embed_dim,  
                                                hungging_model = args.hungging_model, 
                                                fixed_embedding = args.fixed_embedding, 
                                                intra_module_num_layers = args.fine_intra_module_num_layers, 
                                                intra_module_num_heads = args.fine_intra_module_num_heads, 
                                                is_fine = True,  
                                                ) 
        self.mlp_offsets = get_mlp_offset([self.embed_dim, self.embed_dim // 2, 2])
        self.loc_transform = get_mlp([self.embed_dim,self.embed_dim//2,1])
        if args.fine_num_decoder_layers > 0:
            self.cross_hints = nn.ModuleList([nn.TransformerDecoderLayer(d_model = args.fine_embed_dim, 
                                                    nhead = args.fine_num_decoder_heads, 
                                                    dim_feedforward = args.fine_embed_dim * 4) for _ in range(args.fine_num_decoder_layers)])

            self.cross_objects = nn.ModuleList([TransformerDecoderLayer(d_model = args.fine_embed_dim, 
                                                    nhead = args.fine_num_decoder_heads, 
                                                    dim_feedforward = args.fine_embed_dim * 4) for _ in range(args.fine_num_decoder_layers)])
        else:
            self.cross_hints = nn.TransformerDecoderLayer(d_model = args.fine_embed_dim, 
                                                    nhead = args.fine_num_decoder_heads, 
                                                    dim_feedforward = args.fine_embed_dim * 4)
            self.cross_objects = None

        

    def forward(self, objects, hints, object_points):
        batch = []
        batch_size = len(objects)
        num_objects = len(objects[0])
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)
        """
        Textual branch
        """

        hint_encodings = self.language_encoder(hints)

        """
        3D submap branch
        """
        out = self.object_encoder(objects, object_points)
        if type(out) is tuple:
            object_encodings = out[0]
            pos_postions = out[1]
            relative_positions = out[2]

        else:
            object_encodings = out

        relative_positions = relative_positions.to(object_encodings.device)
        object_embeddings = object_encodings.reshape((batch_size, num_objects, self.embed_dim))
        object_embeddings = F.normalize(object_embeddings, dim=-1)
        pos_postions = pos_postions.reshape((batch_size, num_objects, self.embed_dim))
        pos_postions = F.normalize(pos_postions,dim=-1)
        pos_postions = pos_postions.transpose(0,1)
        

        ## CDI Matrix computation: It is recommended to precompute the values and directly retrieve them during training ##
        loc_info_list = []
        for objects_sample in objects:
            for u in range(len(objects_sample)):
                pos0 = objects_sample[u].get_center()
                for v in range(len(objects_sample)):
                    pos1 = objects_sample[v].get_center()
                    direction = get_direction(pos0,pos1)
                    loc_info_list.append(direction)
        loc_info_embeddings = self.language_encoder(loc_info_list).squeeze()
        loc_info_embeddings = self.loc_transform(loc_info_embeddings).squeeze().reshape(batch_size,num_objects,num_objects)
        loc_info_embeddings = F.normalize(loc_info_embeddings,dim=-1).to(object_encodings.device)
        relative_positions = relative_positions + 0.1 * loc_info_embeddings
        relative_positions = F.normalize(relative_positions,dim=-1).repeat(4,1,1).to(object_encodings.device)
        
        """
        CCAT module
        """
        desc0 = object_embeddings.transpose(0, 1)  # [num_obj, B, DIM]
        desc1 = hint_encodings.transpose(0, 1)  # [num_hints, B, DIM]

        if self.cross_objects is not None:
            if len(self.cross_hints) == len(self.cross_objects):
                for i in range(len(self.cross_hints)):
                    desc0 = self.cross_objects[i](desc0, desc1,relative_position=relative_positions,query_pos = pos_postions)
                    desc1 = self.cross_hints[i](desc1, desc0)
            else:
                desc0_new = self.cross_objects[0](desc0, desc1,relative_positions=relative_positions,query_pos = pos_postions)
                desc1_new = self.cross_hints[0](desc1, desc0)
                desc1 = self.cross_hints[1](desc1_new, desc0_new)
        else:
            desc1 = self.cross_hints(desc1, desc0,pos_postions)
            
        desc1 = desc1.max(dim=0)[0]
        offsets = self.mlp_offsets(desc1)


        return offsets

    @property
    def device(self):
        return next(self.mlp_offsets.parameters()).device
    def get_device(self):
        return next(self.mlp_offsets.parameters()).device


def get_pos_in_cell(objects: List[Object3d_K360], matches0, offsets):
    """Extract a pose estimation relative to the cell (∈ [0,1]²) by
    adding up for each matched objects its location plus offset-vector of corresponding hint,
    then taking the average.

    Args:
        objects (List[Object3d_K360]): List of objects of the cell
        matches0 : matches0 from SuperGlue
        offsets : Offset predictions for each hint

    Returns:
        np.ndarray: Pose estimate
    """
    pose_preds = []  # For each match the object-location plus corresponding offset-vector
    for obj_idx, hint_idx in enumerate(matches0):
        if obj_idx == -1 or hint_idx == -1:
            continue
        # pose_preds.append(objects[obj_idx].closest_point[0:2] + offsets[hint_idx]) # Object location plus offset of corresponding hint
        pose_preds.append(
            objects[obj_idx].get_center()[0:2] + offsets[hint_idx]
        )  # Object location plus offset of corresponding hint
    return (
        np.mean(pose_preds, axis=0) if len(pose_preds) > 0 else np.array((0.5, 0.5))
    )  # Guess the middle if no matches


def intersect(P0, P1):
    n = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, np.newaxis]  # normalized
    projs = np.eye(n.shape[1]) - n[:, :, np.newaxis] * n[:, np.newaxis]  # I - n*n.T
    R = projs.sum(axis=0)
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)
    p = np.linalg.lstsq(R, q, rcond=None)[0]
    return p


def get_pos_in_cell_intersect(objects: List[Object3d_K360], matches0, directions):
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    points0 = []
    points1 = []
    for obj_idx, hint_idx in enumerate(matches0):
        if obj_idx == -1 or hint_idx == -1:
            continue
        points0.append(objects[obj_idx].get_center()[0:2])
        points1.append(objects[obj_idx].get_center()[0:2] + directions[hint_idx])
    if len(points0) < 2:
        return np.array((0.5, 0.5))
    else:
        return intersect(np.array(points0), np.array(points1))
