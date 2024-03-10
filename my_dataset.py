#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：ger
@File ：my_dataset.py
@Author ：grey
@Date ：2023/12/20 09:16
'''

import pandas as pd
import numpy as np
import torch
import dgl
from dgl.data import DGLDataset
import json

class MyDataset(DGLDataset):
    def __init__(self, node_path, edge_path):
        super().__init__(name="mydata")
        self.node_path = node_path
        self.edge_path = edge_path

    def process(self):
        nodes_datas = pd.read_csv(self.node_path)
        edges_datas = pd.read_csv(self.edge_path).to_numpy()

        self._make_feature(edges_datas)
        self._get_label(nodes_datas)

    def _make_feature(self, edges_datas):
        self.edges_src_group = list()
        self.edges_des_group = list()
        self.edge_group = list()
        self.node_edge_pair = dict()

        src_index = dict()
        index = -1

        for i in range(len(edges_datas)):
            src, dst, weight = edges_datas[i]
            # demo only
            src = int(src)
            dst = int(dst)
            ########
            src_str = str(src)
            if src_str not in src_index:
                index += 1
                src_index[src_str] = index
                self.node_edge_pair[src_str] = []
                self.edges_src_group.append(list())
                self.edges_des_group.append(list())
                self.edge_group.append(list())

            self.edges_src_group[src_index[src_str]].append(src)
            self.edges_des_group[src_index[src_str]].append(dst)
            self.edge_group[src_index[src_str]].append(weight)
            self.node_edge_pair[src_str].append(dst)
        #
        # print(self.edges_src_group)
        # print(self.edges_des_group)
        # print(self.edge_group)
        # print(self.node_edge_pair)

    def _get_label(self, nodes_datas):
        all_labels = nodes_datas["category"].astype("category").cat.codes.to_numpy()
        all_nodes = nodes_datas["pid"].to_numpy()
        all_weight = nodes_datas["feature"].to_numpy()
        node_relate_dict = dict()
        # 获取对应的点特征和label
        for i in range(len(all_labels)):
            node_id = all_nodes[i]
            __label_feature_pair = list()
            node_relate_dict[str(node_id)] = [json.loads(all_weight[i]),all_labels[i]]

        self.weight_group = list()
        self.node_group = list()
        self.label_group = list()
        self.labels = list()
        for i in range(len(self.edges_src_group)):
            node_id = self.edges_src_group[i][0]
            self.label_group.append(node_relate_dict[str(node_id)][1])
            self.node_group.append([node_id])
            self.weight_group.append([np.array(node_relate_dict[str(node_id)][0])])
            for dst in self.edges_des_group[i]:
                if dst not in self.node_group[i]:
                    self.node_group[i].append(dst)
                    self.weight_group[i].append(np.array(node_relate_dict[str(dst)][0]))
        # print(self.node_group)
        # print(self.label_group)
        # print(self.weight_group)

    def __getitem__(self, idx):
        label = torch.tensor(self.label_group[idx])
        u = torch.from_numpy(np.array(self.edges_src_group[idx]))
        v = torch.from_numpy(np.array(self.edges_des_group[idx]))
        g = dgl.graph((u, v))
        g.ndata['feat'] = torch.from_numpy(np.array(self.weight_group[idx]))
        g.edata['feat'] = torch.from_numpy(np.array(self.edge_group[idx]))

        return label, g

    def __len__(self):
        return len(self.label_group)

dataset = MyDataset()
for i in dataset:
    print(1)




