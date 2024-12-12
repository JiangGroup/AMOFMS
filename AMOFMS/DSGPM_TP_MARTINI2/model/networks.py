import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gnn
from dataset.ham import ATOMS
from dataset.ham import MASK_ATOM_INDEX
NUM_ATOMS = len(ATOMS)


class DSGPM_TP(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(DSGPM_TP, self).__init__()
        self.use_degree_feat = True
        self.use_cycle_feat = True
        self.use_charge_feat = True
        self.use_aromatic_feat = False
        self.bias = True
        self.use_mask_embed = False
        self.num_nn_iter = 6


        self.input_fc, self.nn_conv, self.gru, self.output_fc = self.build_nnconv_layers(input_dim, hidden_dim,
                                                                                         embedding_dim,
                                                                                         layer=gnn.NNConv)

        self.NUM_CG_TYPES = 18
        self.feature_num = embedding_dim + NUM_ATOMS
        if self.use_degree_feat:
            self.feature_num += 1
        if self.use_cycle_feat:
             self.feature_num+= 1
        if self.use_charge_feat:
             self.feature_num+= 1
        if self.use_aromatic_feat:
             self.feature_num+= 1
        # self.cg_type_fc = nn.Linear(self.feature_num, self.NUM_CG_TYPES)
        self.cg_type_fc = nn.Sequential(nn.Linear(self.feature_num, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, self.NUM_CG_TYPES))
        # self.cg_type_fc = nn.Sequential(nn.Linear(self.feature_num, 512),
        #                                 nn.ReLU(),
        #                                 nn.Linear(512, 256),
        #                                 nn.ReLU(),
        #                                 nn.Linear(256, self.NUM_CG_TYPES))

    # input_dim are the total element number
    def build_nnconv_layers(self, input_dim, hidden_dim, embedding_dim, layer=gnn.NNConv):
        if self.use_mask_embed:
            input_fc = nn.Embedding(input_dim + 1, hidden_dim, padding_idx=MASK_ATOM_INDEX)
        else:
            input_fc = nn.Embedding(input_dim, hidden_dim)
        if self.use_degree_feat:
            hidden_dim += 1
        if self.use_cycle_feat:
            hidden_dim += 1
        if self.use_charge_feat:
            hidden_dim += 1
        if self.use_aromatic_feat:
            hidden_dim += 1
        edge_nn = nn.Sequential(
            nn.Linear(4, 128),
            nn.ReLU(),
            nn.Linear(128, hidden_dim*hidden_dim)
        )
        nn_conv = layer(hidden_dim, hidden_dim, edge_nn, aggr='add')
        gru = nn.GRU(hidden_dim, hidden_dim)
        output_fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embedding_dim)
        )
        return input_fc, nn_conv, gru, output_fc

    # def node_bead_type_pred(self, input_dim):

    # x: [natom, 1] tensor, e.g. [[1], [7], [3], [1]]   all value = 1 indicate C (element),
    # all value = 7 indicate S (element), and all value = 3 indicate N (element)

    # edge_index: [2, 2*nbond] tensor, e.g. [[1,2,3,4,5], [2,4,1,5,6]] indicate atom id=1 & 2 are bonded;
    # atom id=2 & 4 are bonded; atom id=3 & 1 are bonded; (double than real bond number)

    # edge_attr:  [2*nbond, 4] one-hot tensor, e.g. [[0, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [1, 0, 0, 0]],
    # there are four types bond, [1, 0, 0, 0] means C-C, [0, 1, 0, 0] means aromatic bond, [0, 0, 1, 0] means c=c,
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        edge_attr = data.edge_attr
        if self.use_cycle_feat or self.use_degree_feat or self.use_charge_feat or self.use_aromatic_feat:
            out = F.relu(self.input_fc(x)).squeeze(1)
            out = torch.cat([out, data.extended_feat], dim=1)
        else:
            out = F.relu(self.input_fc(x)).squeeze(1)
        h = out.unsqueeze(0)

        for i in range(self.num_nn_iter):
            m = F.relu(self.nn_conv(out, edge_index, edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.output_fc(out)

        if self.use_mask_embed:
            atom_types_tensor = torch.zeros((x.shape[0], len(ATOMS) + 1), device=x.device)
        else:
            atom_types_tensor = torch.zeros((x.shape[0], len(ATOMS)), device=x.device)
        atom_types_tensor.scatter_(1, x, 1)

        feat_lst = [out, atom_types_tensor]
        if self.use_cycle_feat or self.use_degree_feat or self.use_charge_feat or self.use_aromatic_feat:
            feat_lst.append(data.extended_feat)
        out = torch.cat(feat_lst, dim=1)
        fg_embed = F.normalize(out)

        node_cg_type_pred = self.cg_type_fc(fg_embed)


        return fg_embed, node_cg_type_pred
