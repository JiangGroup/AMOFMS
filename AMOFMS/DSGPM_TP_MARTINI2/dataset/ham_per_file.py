import os
import glob
import json
import torch
import networkx as nx
import numpy as np
import torch.nn.functional as F

from networkx.algorithms.cycles import cycle_basis
from utils.automorphism_group import node_equal, edge_equal
from torch_geometric.data import Data
from torch.utils.data import Dataset
from .ham import ATOMS, BOND_TYPE_DICT, CG_TYPE_DICT
from rdkit import Chem


class HAMPerFile(Dataset):
    def __init__(self, data_root, cycle_feat=False, degree_feat=False, charge_feat=False, aromatic_feat=False, automorphism=False):
        jsons_root = os.path.join(data_root, '*.json')
        self.json_file_path_lst = glob.glob(jsons_root)
        self.automorphism = automorphism
        self.cycle_feat = cycle_feat
        self.degree_feat = degree_feat
        self.charge_feat = charge_feat
        self.aromatic_feat = aromatic_feat

    def __getitem__(self, index):
        """
        get index-th data
        :param index: index from lst
        :return: atom_types, fg_adj, cg_adj_gt, mapping_op_gt, atomic_weight
        """
        json_fpath = self.json_file_path_lst[index]

        with open(json_fpath) as f:
            json_data = json.load(f)
        """
            json_data format:
            dict {
                "cgnodes": [],  # [[fg_id...], [fg_id...]],
                "nodes": [
                            {
                                "cg":2,  # cg group_id (starts with 0)
                                "element":"C",  # atom type
                                "id":0  # fg id
                            },
                            {...}
                         ],
                "edges": [
                            {
                                "source":0,  # from fg_id
                                "target":1   # to fg_id
                                "bondtype": 1.0  # bond type (1.0, 1.5, 2.0, 3.0)
                            }
                         ],
                "smiles": "C[SiH](C)O[Si](C)(CCl)F"
            }
        """
        data = Data()

        if 'smiles' not in json_data:
            smiles = os.path.splitext(os.path.basename(json_fpath))[0]
        else:
            smiles = json_data['smiles']
        smiles = smiles.replace('/', '\\')
        graph = nx.Graph(smiles=smiles)
        for node in json_data['nodes']:
            graph.add_node(node['id'], element=node['element'], cg=node['cg_id'])

        for edge in json_data['edges']:
            bond_type = edge['bondtype']
            if isinstance(bond_type, str):
                assert bond_type in {'-', '='}
                bond_type = {'-': 1.0, '=': 2.0}
            graph.add_edge(edge['source'], edge['target'], bond_type=bond_type)

        # ========== load atom types ==========
        fg_beads: list = json_data['nodes']
        fg_beads.sort(key=lambda x: x['id'])
        # atom_types = torch.LongTensor([ATOM_TYPES.index(bead['element']) for bead in fg_beads]).reshape(-1, 1)
        atom_types = torch.LongTensor([list(ATOMS.keys()).index(bead['element']) for bead in fg_beads]).reshape(-1, 1)
        data.x = atom_types

        # ========== load atom charges ==========
        data.atom_charges = torch.Tensor([[bead['charge']] for bead in fg_beads]).reshape(-1, 1)

        # ========== load atom aromatic properties ==========
        mol = Chem.MolFromSmiles(smiles)
        atom_aromatics = []
        for atom in mol.GetAtoms():
            atom_index = atom.GetIdx()
            is_aromatic = atom.GetIsAromatic()
            atom_aromatics.append({'id': atom_index, 'IsAromatics':int(is_aromatic)})
        atom_aromatics.sort(key=lambda x: x['id'])
        data.atom_aromatics = torch.Tensor([[bead['IsAromatics']] for bead in atom_aromatics]).reshape(-1, 1)

        # ========== load CG types ==========
        # print(json_data['smiles'])
        # atom_CG_types = torch.LongTensor([list(CG_TYPE_DICT.keys()).index(bead['cg_type']) for bead in fg_beads]).reshape(-1, 1)
        # data.atom_CG_types = atom_CG_types

        # ======== degree ===========
        if self.degree_feat:
            degrees = graph.degree
            degrees = np.array(degrees)[:, 1]
            degrees = torch.tensor(degrees).float().unsqueeze(dim=-1) / 4
            if hasattr(data, 'extended_feat'):
                data.extended_feat = torch.cat([data.extended_feat, degrees], dim=1)
            else:
                data.extended_feat = degrees

        # ========= cycles ==========
        if self.cycle_feat:
            cycle_indicator_per_node = torch.zeros(len(fg_beads)).unsqueeze(-1)
            cycle_lst = cycle_basis(graph)
            if len(cycle_lst) > 0:
                for idx_cycle, cycle in enumerate(cycle_lst):
                    cycle = torch.tensor(cycle)
                    cycle_indicator_per_node[cycle] = 1
            if hasattr(data, 'extended_feat'):
                data.extended_feat = torch.cat([data.extended_feat, cycle_indicator_per_node], dim=1)
            else:
                data.extended_feat = cycle_indicator_per_node

        # ========= charge ==========
        if self.charge_feat:
            if hasattr(data, 'extended_feat'):
                data.extended_feat = torch.cat([data.extended_feat, data.atom_charges], dim=1)
            else:
                data.extended_feat = data.atom_charges

        # ========= aromatic ==========
        if self.aromatic_feat:
            if hasattr(data, 'extended_feat'):
                data.extended_feat = torch.cat([data.extended_feat, data.atom_aromatics], dim=1)
            else:
                data.extended_feat = data.atom_aromatics

        edges = []
        bond_types = []
        for x in json_data['edges']:
            edges.append([x['source'], x['target']])
            edges.append([x['target'], x['source']])
            bond_types.append(BOND_TYPE_DICT[x['bondtype']])
            bond_types.append(BOND_TYPE_DICT[x['bondtype']])  # add bond types for both directions
        data.edge_index = torch.tensor(edges).long().t()
        data.no_bond_edge_attr = torch.ones(data.edge_index.shape[1])
        data.edge_attr = F.one_hot(torch.tensor(bond_types, dtype=torch.long), num_classes=4).float()
        assert data.edge_attr.shape == (len(bond_types), 4)

        # ========== load ground truth ==========
        assert len(atom_types) == len(json_data['nodes'])
        original_mapping = self.compute_cluster_idx(json_data)
        if self.automorphism:
            gm = nx.isomorphism.GraphMatcher(graph, graph,
                                             node_match=node_equal,
                                             edge_match=edge_equal)
            mapping_lst = []
            for node_mapping in gm.isomorphisms_iter():
                key_value_lst = torch.tensor(list(node_mapping.items())).transpose(1, 0)
                new_mapping = original_mapping.clone()
                new_mapping[key_value_lst[0]] = new_mapping[key_value_lst[1]]
                mapping_lst.append(new_mapping)
            data.y = torch.stack(mapping_lst)
        else:
            data.y = original_mapping

        data.graph = graph
        data.json = json_data
        data.fname = os.path.splitext(os.path.basename(json_fpath))[0]

        return data

    def __len__(self):
        return len(self.json_file_path_lst)

    @staticmethod
    def compute_cluster_idx(json_data):
        node_cluster_index = -1 * torch.ones((len(json_data['nodes']),)).long()
        for node in json_data['nodes']:
            fg_id, cg_id = node['id'], node['cg_id']
            node_cluster_index[fg_id] = cg_id
        assert torch.all(node_cluster_index >= 0)
        return node_cluster_index
