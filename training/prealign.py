import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import collections
from typing import List
import torch_geometric.transforms as T
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from easydict import EasyDict
import os
import os.path as osp
import tqdm
import cv2
from training.args import parse_arguments
from datapreparation.kitti360pose.utils import (
    COLORS as COLORS_K360,
    COLOR_NAMES as COLOR_NAMES_K360,
    SCENE_NAMES_TEST,
)
from dataloading.kitti360pose.poses import Kitti360FineDataset, Kitti360FineDatasetMulti
from datapreparation.kitti360pose.utils import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
# from models.language_encoder import get_mlp, LanguageEncoder
# from models.object_encoder import ObjectEncoder
from models.fine.language_encoder import get_mlp
from models.pointcloud.pointnet2 import PointNet2

from datapreparation.kitti360pose.imports import Object3d
from datapreparation.kitti360pose.utils import COLOR_NAMES
from nltk import tokenize as text_tokenize
# if nltk not work well add the following command
# nltk.download('punkt')
from transformers import AutoTokenizer, T5EncoderModel
def get_mlp2(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct and MLP for use in other models without RELU in the final layer.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                ) if i < len(channels) - 1
                else
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i])
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                if i < len(channels) - 1
                else nn.Sequential(nn.Linear(channels[i - 1], channels[i]))
                for i in range(1, len(channels))
            ]
        )

class LanguageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim,  hungging_model = None, fixed_embedding=False, 
                 intra_module_num_layers=2, intra_module_num_heads=4, 
                 is_fine = False, inter_module_num_layers=2, inter_module_num_heads=4,
                 ):
        """Language encoder to encode a set of hints for each sentence"""
        super(LanguageEncoder, self).__init__()

        self.is_fine = is_fine
        self.tokenizer = AutoTokenizer.from_pretrained("PATH_TO_T5")
        T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
        self.llm_model = T5EncoderModel.from_pretrained("PATH_TO_T5")
        if fixed_embedding:
            self.fixed_embedding = True
            for para in self.llm_model.parameters():
                para.require_grads = False
        else:
            self.fixed_embedding = False

        input_dim = self.llm_model.encoder.embed_tokens.weight.shape[-1]

        self.intra_module = nn.ModuleList([nn.TransformerEncoderLayer(input_dim, intra_module_num_heads,  dim_feedforward = input_dim * 4) for _ in range(intra_module_num_layers)])

        self.inter_mlp = get_mlp2([input_dim, embedding_dim], add_batchnorm=True)
        
        # if not is_fine:
        #     self.inter_module = nn.ModuleList([nn.TransformerEncoderLayer(embedding_dim, inter_module_num_heads,  dim_feedforward = embedding_dim * 4) for _ in range(inter_module_num_layers)])
            
    
    def forward(self, descriptions):

        split_union_sentences = []
        for description in descriptions:
            split_union_sentences.extend(text_tokenize.sent_tokenize(description))

        
        batch_size = len(descriptions)
        num_sentence = len(split_union_sentences) // batch_size

        inputs = self.tokenizer(split_union_sentences, return_tensors="pt", padding = "longest")
        shorten_sentences_indices = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        shorten_sentences_indices = shorten_sentences_indices.to(self.device)
        attention_mask = attention_mask.to(self.device)
        out = self.llm_model(input_ids = shorten_sentences_indices, 
                        attention_mask = attention_mask,
                        output_attentions = False)
        description_encodings = out.last_hidden_state
        
        if self.fixed_embedding:
            description_encodings = description_encodings.detach()
        description_encodings = description_encodings.max(dim = 1)[0]

        description_encodings = self.inter_mlp(description_encodings)
        return description_encodings

    @property
    def device(self):
        return next(self.inter_mlp.parameters()).device
    

class ObjectEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int, known_classes: List[str], known_colors: List[str], args):
        """Module to encode a set of instances (objects / stuff)

        Args:
            embed_dim (int): Embedding dimension
            known_classes (List[str]): List of known classes
            known_colors (List[str]): List of known colors
            args: Global training arguments
        """
        super(ObjectEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.args = args

        # Set idx=0 for padding
        self.known_classes = {c: (i + 1) for i, c in enumerate(known_classes)}
        self.known_classes["<unk>"] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)

        self.known_colors = {c: i for i, c in enumerate(COLOR_NAMES)}
        self.known_colors["<unk>"] = 0
        self.color_embedding = nn.Embedding(len(self.known_colors), embed_dim, padding_idx=0)

        self.pos_encoder = get_mlp([3, 64, embed_dim])  # OPTION: pos_encoder layers
        self.color_encoder = get_mlp([3, 64, embed_dim])  # OPTION: color_encoder layers
        self.num_encoder = get_mlp([1, 64, embed_dim])

        self.num_mean = 1826.6844940968194
        self.num_std = 2516.8905096993817
        

        self.pointnet = PointNet2(
            len(known_classes), len(known_colors), args
        )  # The known classes are all the same now, at least for K360
        self.pointnet_dim = self.pointnet.lin2.weight.size(0)

        if args.pointnet_freeze:
            print("CARE: freezing PN")
            self.pointnet.requires_grad_(False)
        

        if args.pointnet_features == 0:
            self.mlp_pointnet = get_mlp([self.pointnet.dim0, self.embed_dim])
        elif args.pointnet_features == 1:
            self.mlp_pointnet = get_mlp([self.pointnet.dim1, self.embed_dim])
        elif args.pointnet_features == 2:
            self.mlp_pointnet = get_mlp([self.pointnet.dim2, self.embed_dim])
        self.mlp_merge = get_mlp([len(args.use_features) * embed_dim, embed_dim])

    def forward(self, objects: List[Object3d], object_points):
        """Features are currently normed before merging but not at the end.

        Args:
            objects (List[List[Object3d]]): List of lists of objects
            object_points (List[Batch]): List of PyG-Batches of object-points
        """

        if ("class_embed" in self.args and self.args.class_embed) or (
            "color_embed" in self.args and self.args.color_embed
        ):
            class_indices = []
            color_indices = []
            for i_batch, objects_sample in enumerate(objects):
                for obj in objects_sample:
                    class_idx = self.known_classes.get(obj.label, 0)
                    class_indices.append(class_idx)
                    color_idx = self.known_colors[obj.get_color_text()]
                    color_indices.append(color_idx)

        if "class_embed" not in self.args or self.args.class_embed == False:
            # Void all colors for ablation
            if "color" not in self.args.use_features:
                for pyg_batch in object_points:
                    pyg_batch.x[:] = 0.0  # x is color, pos is xyz

            object_features = [
                self.pointnet(pyg_batch.to(self.get_device())).features2
                for pyg_batch in object_points
            ]  # [B, obj_counts, PN_dim]


            object_features = torch.cat(object_features, dim=0)  # [total_objects, PN_dim]
            object_features = self.mlp_pointnet(object_features)

        embeddings = []
        if "class" in self.args.use_features:
            if (
                "class_embed" in self.args and self.args.class_embed
            ):  # Use fixed embedding (ground-truth data!)
                class_embedding = self.class_embedding(
                    torch.tensor(class_indices, dtype=torch.long, device=self.get_device())
                )
                embeddings.append(F.normalize(class_embedding, dim=-1))
            else:
                embeddings.append(
                    F.normalize(object_features, dim=-1)
                )  # Use features from PointNet

        if "color" in self.args.use_features:
            if "color_embed" in self.args and self.args.color_embed:
                color_embedding = self.color_embedding(
                    torch.tensor(color_indices, dtype=torch.long, device=self.get_device())
                )
                embeddings.append(F.normalize(color_embedding, dim=-1))
            else:
                colors = []
                for objects_sample in objects:
                    colors.extend([obj.get_color_rgb() for obj in objects_sample])
                color_embedding = self.color_encoder(
                    torch.tensor(colors, dtype=torch.float, device=self.get_device())
                )
                embeddings.append(F.normalize(color_embedding, dim=-1))

        if "position" in self.args.use_features:
            positions = []
            for objects_sample in objects:
                positions.extend([obj.get_center() for obj in objects_sample])
            pos_positions = torch.tensor(positions, dtype=torch.float, device=self.get_device())
            pos_embedding = self.pos_encoder(pos_positions)
            embeddings.append(F.normalize(pos_embedding, dim=-1))

        if "num" in self.args.use_features:
            num_points = []
            for objects_sample in objects:
                num_points.extend([len(obj.xyz) for obj in objects_sample])
            num_points_embedding = self.num_encoder(
                (torch.tensor(num_points, dtype=torch.float, device=self.get_device()).unsqueeze(-1) - self.num_mean) / self.num_std
            )
            embeddings.append(F.normalize(num_points_embedding, dim=-1))


        if len(embeddings) > 1:
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]

        return embeddings, pos_positions,F.normalize(object_features, dim=-1),F.normalize(color_embedding, dim=-1)

    @property
    def device(self):
        return next(self.class_embedding.parameters()).device

    def get_device(self):
        return next(self.class_embedding.parameters()).device
    
class pretrain_object(nn.Module):
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
        super(pretrain_object, self).__init__()
        self.embed_dim = args.fine_embed_dim
        self.object_encoder = ObjectEncoder(args.fine_embed_dim, known_classes, known_colors, args)
        self.language_encoder = LanguageEncoder(args.fine_embed_dim,  
                                                hungging_model = args.hungging_model, 
                                                fixed_embedding = args.fixed_embedding, 
                                                intra_module_num_layers = args.fine_intra_module_num_layers, 
                                                intra_module_num_heads = args.fine_intra_module_num_heads, 
                                                is_fine = True,  
                                                ) 
    
    def forward(self,objects,object_points):

        embeddings, pos_positions,object_feature,color_feature = self.object_encoder(objects,object_points)
        label_embed = []
        color_embed = []
        for i in range(len(objects)):
            color_list = []
            label_list = []
            object_list  =  objects[i]
            for n in range(len(object_list)):
                color = object_list[n].get_color_text()
                label = object_list[n].label
                color_list.append(f"{color}")
                label_list.append(f"{label}")
            label_feat = self.language_encoder(label_list)
            color_feat = self.language_encoder(color_list)
            label_embed.append(label_feat)
            color_embed.append(color_feat)
        t5_label_embed = torch.cat(label_embed,dim=0)
        t5_color_embed = torch.cat(color_embed,dim=0)
        return object_feature,color_feature,t5_label_embed,t5_color_embed
    
def train_epoch(model,dataloader,args):
    model.train()
    stats = EasyDict(
        loss=[],
    )
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))
    for i_batch, batch in pbar:
        optimizer.zero_grad()
        object_feature,color_feature,t5_label_embed,t5_color_embed = model(batch["objects"],batch["object_points"])
        loss = loss_label(object_feature,t5_label_embed) + loss_color(color_feature,t5_color_embed)
        loss.backward()
        optimizer.step()
        stats.loss.append(loss.item())
        pbar.set_postfix(loss_color = loss_color(color_feature,t5_color_embed).item(),loss_label = loss_label(object_feature,t5_label_embed).item(),loss = loss.item())
    for key in stats.keys():
        stats[key] = np.mean(stats[key])
    return stats

def seed_everything(seed: int): 
   random.seed(seed) 
   os.environ['PYTHONHASHSEED'] = str(seed) 
   np.random.seed(seed) 
   torch.manual_seed(seed) 
   torch.cuda.manual_seed(seed) 
   torch.backends.cudnn.deterministic = True 
   torch.backends.cudnn.benchmark = True 



if __name__ == "__main__":
    seed_everything(42)
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    dataset_name = args.base_path[:-1] if args.base_path.endswith("/") else args.base_path
    dataset_name = dataset_name.split("/")[-1]
    print(f"Directory: {dataset_name}")

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    folder_name = args.folder_name
    print("#####################")
    print("########   Folder Name: " + folder_name)
    print("#####################")
    # if not osp.isdir(f"./checkpoints/{dataset_name}/{folder_name}"):
    #     os.mkdir(f"./checkpoints/{dataset_name}/{folder_name}")

    """
    Create data loaders
    """
    if args.dataset == "K360":
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360FineDatasetMulti(
            args.base_path, SCENE_NAMES_TRAIN, train_transform, args, flip_pose=False,
            pmc_prob = args.pmc_prob,
            pmc_threshold = args.pmc_threshold,
        ) 
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360FineDataset.collate_fn,
            shuffle=args.shuffle,
        )

        dataset_val = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform, args,)
        dataloader_val = DataLoader(
            dataset_val, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn
        )

        dataset_test = Kitti360FineDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform, args,)
        dataloader_test = DataLoader(
            dataset_test, batch_size=args.batch_size, collate_fn=Kitti360FineDataset.collate_fn
        )

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data0 = dataset_train[0]
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)
    model = pretrain_object(dataset_train.get_known_classes(),COLOR_NAMES_K360,args)
    model.to(device)
    loss_label = nn.MSELoss()
    loss_color = nn.MSELoss()
    # model_dic = model.state_dict()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    lr = args.learning_rate
    num_epoch_warmup = 3
    for epoch in range(1, args.epochs + 1):
        if epoch == num_epoch_warmup:
            optimizer = optim.Adam(model.parameters(), lr=lr)
            if args.lr_scheduler == "exponential":
                scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
            elif args.lr_scheduler == "step":
                scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
            else:
                raise TypeError

        train_out = train_epoch(model, dataloader_train, args,)
        if scheduler:
            scheduler.step()

        print(
            (
                f"\t lr {lr:0.6} epoch {epoch} loss {train_out.loss:0.3f} "
            ),
            flush=True,
        )
        best_loss = 1 


        loss = np.mean(train_out.loss)

        if loss < best_loss:
            pointnet_path = f"./checkpoints/{dataset_name}/{folder_name}/epoch{epoch}_pointnet.pth"
            color_path = f"./checkpoints/{dataset_name}/{folder_name}/epoch{epoch}_color_encoder.pth"
            mlp_path = f"./checkpoints/{dataset_name}/{folder_name}/epoch{epoch}_mlp.pth"
            
            model_dic = model.state_dict()
                # out = collections.OrderedDict()
                # for item in model_dic:
                #     if "llm_model" not in item:
                #         out[item] = model_dic[item]
            pointnet_dict = {k:v for k,v in model_dic.items() if "pointnet" in k}
            color_dict = {k:v for k,v in model_dic.items() if "color_encoder" in k}
            mlp_dict = {k:v for k,v in model_dic.items() if "inter_mlp" in k}
            torch.save(pointnet_dict, pointnet_path)
            torch.save(color_dict, color_path)
            torch.save(mlp_dict, mlp_path)
                
            best_loss = loss

