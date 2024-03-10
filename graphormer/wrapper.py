# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import random
import torch
import numpy as np
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
import pandas as pd
import random
from torch_geometric.data import Data, InMemoryDataset
import json

pyximport.install()
import algos


def convert_to_single_emb(x, offset=512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x.long()


def preprocess_item(item, idx):

    num_virtual_tokens = 1
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    if edge_attr is None:
        edge_attr = torch.zeros((edge_index.shape[1]), dtype=torch.long)

    N = x.size(0)

    #x = convert_to_single_emb(x)  # For ZINC: [n_nodes, 1] 修改

    # node adj matrix [N, N] bool
    adj_orig = torch.zeros([N, N], dtype=torch.bool)
    adj_orig[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here

    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )  # [n_nodes, n_nodes, 1] for ZINC

    shortest_path_result, path = algos.floyd_warshall(
        adj_orig.numpy()
    )  # [n_nodesxn_nodes, n_nodesxn_nodes]
    max_dist = np.amax(shortest_path_result)
    edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy().astype(int))
    rel_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.float
    )  # with graph token

    adj = torch.zeros(
        [N + num_virtual_tokens, N + num_virtual_tokens], dtype=torch.bool
    )
    adj[edge_index[0, :], edge_index[1, :]] = True

    for i in range(num_virtual_tokens):
        adj[N + i, :] = True
        adj[:, N + i] = True

    # for i in range(N + num_virtual_tokens):
    #     for j in range(N + num_virtual_tokens):

    #         val = True if random.random() < 0.3 else False
    #         adj[i, j] = adj[i, j] or val

    # combine
    item.idx = idx
    item.x = x
    item.adj = adj
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.rel_pos = rel_pos
    item.in_degree = adj_orig.long().sum(dim=1).view(-1)
    item.out_degree = adj_orig.long().sum(dim=0).view(-1)
    item.edge_input = torch.from_numpy(edge_input).long()
    item.adj = adj

    return item


class MyGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        super(MyGraphPropPredDataset, self).download()

    def process(self):
        super(MyGraphPropPredDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyPygPCQM4MDataset(PygPCQM4MDataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyZINCDataset(torch_geometric.datasets.ZINC):
    def download(self):
        super(MyZINCDataset, self).download()

    def process(self):
        super(MyZINCDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


class MyCoraDataset(torch_geometric.datasets.Planetoid):
    def download(self):
        super(MyCoraDataset, self).download()

    def process(self):
        super(MyCoraDataset, self).process()

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


"""
class MyDataset(InMemoryDataset):
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
"""

#重写了一下自己的dataset
"""
class MyDataSet(InMemoryDataset):
    def __init__(self, root, name, feature_size, edge_path, node_path, transform=None, pre_transform=None):
        self.edge_path = edge_path
        self.node_path = node_path
        self.feature_size = feature_size
        print(f'feature size: {feature_size}')

        super(MyDataSet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['my_dataset.pt',]

    @property
    def processed_file_names(self):
        return ['my_dataset.pt',]

    def download(self):
        pass

    def process(self):
        nodes_datas = pd.read_csv(self.node_path)
        edges_datas = pd.read_csv(self.edge_path).to_numpy()
        node_ids = nodes_datas['pid'].values
        new_ids = np.arange(len(node_ids)).astype(int)
        dict_for_ids = {}
        for i in range(len(node_ids)):
            if node_ids[i] not in dict_for_ids.keys():
                dict_for_ids[node_ids[i]] = new_ids[i]

        features_raw = nodes_datas['feature'].values
        features = []
        for i in range(len(features_raw)):
            line = features_raw[i][2:-2]  #去掉"[]"
            line = line.split(',')
            line = torch.from_numpy(np.array(list(map(float, line))))
            features.append(line)

        features = torch.stack(features, dim=0) #获取节点特征
        #print(features)
        # 邻接矩阵
        network = []
        edge_attrs = []
        for i in range(len(edges_datas)):
            src, dst, weight = edges_datas[i]
            #print(dst)
            link = [int(dict_for_ids[src]), int(dict_for_ids[dst])]
            link = torch.from_numpy(np.array(link)).long()
            edge_attrs.append(weight)
            network.append(link)


        network = torch.stack(network, dim=0).long()
        edge_attrs = torch.from_numpy(np.array(edge_attrs))
        # do processing, get x, y, edge_index ready.
        labels = torch.from_numpy(nodes_datas['category'].values)
        graph = Data(x=features, edge_index=network.t().contiguous(), edge_attr=edge_attrs, y=labels)

        #train idx, val idx, test idx
        new_ids_lst = new_ids.tolist()
        lenth_ids = len(new_ids_lst)
        random.seed(42)
        random.shuffle(new_ids_lst)
        train_idx = torch.zeros(lenth_ids).bool()
        train_idx[new_ids_lst[:int(lenth_ids*0.2)]] = True
        #print(train_idx)
        val_idx = torch.zeros(lenth_ids).bool()
        val_idx[new_ids_lst[int(lenth_ids*0.2):int(lenth_ids*0.5)]] = True
        test_idx = torch.zeros(lenth_ids).bool()
        test_idx[new_ids_lst[int(lenth_ids*0.5):]] = True
        #val_idx = torch.from_numpy(np.array(new_ids_lst[int(lenth_ids*0.2):int(lenth_ids*0.5)])).long()
        #test_idx = torch.from_numpy(np.array(new_ids_lst[int(lenth_ids*0.5):])).long()
        #train_idx = torch.tensor([id2inter_id[idx] for idx in herb_with_label_id], dtype=torch.long)
        #加入新的属性
        graph.train_mask = train_idx
        graph.val_mask = val_idx
        graph.test_mask = test_idx

        if self.pre_filter is not None:
            graph = [data for data in graph if self.pre_filter(data)]

        if self.pre_transform is not None:
            graph = [self.pre_transform(data) for data in graph]

        data, slices = self.collate([graph])
        torch.save((data, slices), self.processed_paths[0])

    def __getitem__(self, idx):
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item.idx = idx
            return preprocess_item(item)
        else:
            return self.index_select(idx)


if __name__ == "__main__":
    data = MyDataSet(
        name="MY_DATA", root="../custom_dataset/", feature_size=128,
        edge_path=r"D:\PycharmProjects\Graph_transformer\own_data\edge.csv",
        node_path=r"D:\PycharmProjects\Graph_transformer\own_data\node.csv"
    )
    print(data.data)
"""

class MyDataSet(InMemoryDataset):
    def __init__(self, root, name, feature_size, edge_path, node_path, transform=None, pre_transform=None):
        self.edge_path = edge_path
        self.node_path = node_path
        self.feature_size = feature_size
        print(f'feature size: {feature_size}')

        super(MyDataSet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])


    @property
    def raw_file_names(self):
        return ['my_dataset.pt', ]

    @property
    def processed_file_names(self):
        return ['my_dataset.pt', ]

    def download(self):
        pass

    def process(self):
        nodes_datas = pd.read_csv(self.node_path)
        edges_datas = pd.read_csv(self.edge_path)
        node_ids = nodes_datas['pid'].values
        new_ids = np.arange(len(node_ids)).astype(int)
        nodes_datas['pid'] = new_ids
        dict_for_ids = {}
        for i in range(len(node_ids)):
            if node_ids[i] not in dict_for_ids.keys():
                dict_for_ids[node_ids[i]] = new_ids[i]

        src_values = edges_datas['src'].values
        dst_values = edges_datas['dst'].values
        for i in range(len(src_values)):
            src_values[i] = dict_for_ids[src_values[i]]
            dst_values[i] = dict_for_ids[dst_values[i]]

        src_values = src_values.astype(int)
        dst_values = dst_values.astype(int)
        edges_datas['src'] = src_values
        edges_datas['dst'] = dst_values

        edges_datas = edges_datas.to_numpy()

        self._make_feature(edges_datas)
        self._get_label(nodes_datas)

        self.graphs = []

        for idx in range(len(self.weight_group)):
            edge_indexes = []
            edge_attr = []
            features = torch.from_numpy(np.array(self.weight_group[idx]))
            """
            u = torch.from_numpy(np.array(self.edges_src_group[idx]))
            u = torch.div(u, 1)
            v = torch.from_numpy(np.array(self.edges_des_group[idx]))
            v = torch.div(v, 1)
            print(u)
            print(v)
            """
            for j in range(features.shape[0]-1):
                u = torch.add(torch.zeros(features.shape[0]-1-j), j)
                nodes_num = torch.arange(j, features.shape[0]-1)
                v = torch.add(nodes_num, 1)
            #解决边对不上的问题
                edge_indexes.append(torch.stack([u, v], dim=0).long())
                edge_attr.append(torch.from_numpy(np.array(self.edge_group[idx+j])))
            edge_indexes = torch.cat(edge_indexes, dim=1)
            edge_attr = torch.cat(edge_attr)
            label = torch.tensor(self.label_group[idx]).long()
            g = Data(x=features, edge_index=edge_indexes, edge_attr=edge_attr, y=label)

            self.graphs.append(g)

        if self.pre_filter is not None:
            self.graphs = [data for data in self.graphs if self.pre_filter(data)]

        if self.pre_transform is not None:
            self.graphs = [self.pre_transform(data) for data in self.graphs]

        data, slices = self.collate(self.graphs)
        torch.save((data, slices), self.processed_paths[0])

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
            #src_str = str(src)
            if src not in src_index:
                index += 1
                src_index[src] = index
                self.node_edge_pair[src] = []
                self.edges_src_group.append(list())
                self.edges_des_group.append(list())
                self.edge_group.append(list())

            self.edges_src_group[src_index[src]].append(src)
            self.edges_des_group[src_index[src]].append(dst)
            self.edge_group[src_index[src]].append(weight)
            self.node_edge_pair[src].append(dst)

    def _get_label(self, nodes_datas):
        all_labels = nodes_datas["category"].astype("category").cat.codes.to_numpy()
        all_nodes = nodes_datas["pid"].to_numpy()
        all_weight = nodes_datas["feature"].to_numpy()
        node_relate_dict = dict()
        # 获取对应的点特征和label
        for i in range(len(all_labels)):
            node_id = all_nodes[i]
            __label_feature_pair = list()
            node_relate_dict[node_id] = [json.loads(all_weight[i]), all_labels[i]]

        self.weight_group = list()
        self.node_group = list()
        self.label_group = list()
        self.labels = list()
        for i in range(len(self.edges_src_group)):
            node_id = self.edges_src_group[i][0]
            self.label_group.append(node_relate_dict[node_id][1])
            self.node_group.append([node_id])
            self.weight_group.append([np.array(node_relate_dict[node_id][0])])
            for dst in self.edges_des_group[i]:
                if dst not in self.node_group[i]:
                    self.node_group[i].append(dst)
                    self.weight_group[i].append(np.array(node_relate_dict[dst][0]))
        # print(self.node_group)
        # print(self.label_group)
        # print(self.weight_group)

    def __getitem__(self, idx):
        #label = torch.tensor(self.label_group[idx])
        #u = torch.from_numpy(np.array(self.edges_src_group[idx]))
        #v = torch.from_numpy(np.array(self.edges_des_group[idx]))
        #g = dgl.graph((u, v))
        #g.ndata['feat'] = torch.from_numpy(np.array(self.weight_group[idx]))
        #g.edata['feat'] = torch.from_numpy(np.array(self.edge_group[idx]))
        if isinstance(idx, int):
            item = self.get(self.indices()[idx])
            item = preprocess_item(item, idx)
        else:
            item = self.index_select(idx)

        return item

    #def __len__(self):
        #return len(self.graphs)

if __name__ == "__main__":
    data = MyDataSet(
        name="MY_DATA", root=r"D:/PycharmProjects/Graph_transformer/custom_dataset/", feature_size=129,
        edge_path=r"D:\PycharmProjects\Graph_transformer\own_data\edge.csv",
        node_path=r"D:\PycharmProjects\Graph_transformer\own_data\node.csv"
    )
    for i in range(len(data)):
        print(data[i].edge_index)








