import torch
import tqdm
import numpy as np
import os
import json
import metis
import copy

from dataset.ham_per_file import HAMPerFile
from option import arg_parse
from model.networks import DSGPM_TP
from torch_geometric.data import DataListLoader
from model.graph_cuts import graph_cuts
from utils.post_processing import enforce_connectivity
from model.graph_cuts import graph_cuts_with_adj
from torch_geometric.nn.pool import graclus
import torch.nn.functional as F

from warnings import simplefilter
from sklearn.exceptions import UndefinedMetricWarning
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)
from collections import Counter
import random


def adjust_list(lst):
    counts = Counter(lst)
    max_count = counts.most_common(1)[0][1]
    most_common_elements = [element for element, count in counts.items() if count == max_count]
    chosen_element = random.choice(most_common_elements)
    return [chosen_element] * len(lst)


CG_TYPE_DICT = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4, 'NDA': 5, 'ND': 6, 'NA': 7, 'N0': 8, 'P1': 9, 'P2': 10,
                'P3': 11, 'P4': 12, 'P5': 13, 'QDA': 14, 'QD': 15, 'QA': 16, 'Q0': 17}
CG_TYPE_DICT = {value: key for key, value in CG_TYPE_DICT.items()}

def eval(test_dataloader, model, args):
    model.eval()

    tbar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), dynamic_ncols=True)
    for i, data in tbar:
        data = data[0]
        json_data = data.json
        json_data['cgnodes'] = []
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        data = data.to(torch.device(0))
        edge_index_cpu = data.edge_index.cpu().numpy()
        fg_embed, node_cg_type_pred = model(data)
        softmax_output = F.softmax(node_cg_type_pred, dim=1)
        predicted_cg_types_id = torch.argmax(softmax_output.cpu(), dim=1)
        predicted_cg_types = [CG_TYPE_DICT[cgtype.item()] for cgtype in predicted_cg_types_id]

        dense_adj = torch.sparse.LongTensor(data.edge_index, data.no_bond_edge_attr, (num_nodes, num_nodes)).to_dense()

        if args.num_cg_beads is None:
            iter_num_cg_beads = range(2, num_nodes)
        else:
            iter_num_cg_beads = args.num_cg_beads

        for num_cg_bead in iter_num_cg_beads:
            # try:
            if args.inference_method == 'dsgpm_tp':
                hard_assign, _ = graph_cuts(fg_embed, data.edge_index, num_cg_bead, args.bandwidth, device=args.device_for_affinity_matrix)
                hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
            elif args.inference_method == 'spec_cluster':
                hard_assign = graph_cuts_with_adj(dense_adj, num_cg_bead)
                hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
            elif args.inference_method == 'metis':
                hard_assign = metis.part_graph(data.graph, nparts=num_cg_bead)[1]
            elif args.inference_method == 'graclus':
                hard_assign = graclus(data.edge_index.cpu()).cpu()
                hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
            actual_num_cg = max(hard_assign) + 1
            if actual_num_cg != num_cg_bead:
                print('warning: actual vs. expected: {} vs. {}'.format(actual_num_cg, num_cg_bead))

            result_json = copy.deepcopy(json_data)
            for atom_idx, cg_idx in enumerate(hard_assign):
                result_json['nodes'][atom_idx]['cg_id'] = int(cg_idx)
                result_json['nodes'][atom_idx]['cg_type'] = predicted_cg_types[atom_idx]
            result_json['cgnode_types'] = predicted_cg_types

            for cg_idx in range(num_cg_bead):
                atom_indices = np.nonzero(hard_assign == cg_idx)[0].tolist()
                atom_indices = [int(x) for x in atom_indices]
                result_json['cgnodes'].append(atom_indices)

            if args.use_regular_mapping_from_prediction:
                cg_groups = []
                for i in range(num_cg_bead):
                    cg_groups_tmp = []
                    for id, cg in enumerate(hard_assign):
                        if i == cg:
                            cg_groups_tmp.append(predicted_cg_types[id])
                    cg_groups.append(cg_groups_tmp)

                new_cg_groups = [adjust_list(sublist) for sublist in cg_groups]


                new_predicted_cg_types = [new_cg_groups[i][0] for i in hard_assign]
                # new_predicted_cg_types = [item for sublist in new_cg_groups for item in sublist]
                for atom_idx, cg_idx in enumerate(hard_assign):
                    result_json['nodes'][atom_idx]['cg_type'] = new_predicted_cg_types[atom_idx]
                result_json['cgnode_types'] = new_predicted_cg_types
                # print(hard_assign)
                # print(predicted_cg_types)
                # print(new_predicted_cg_types)

            fpath = os.path.join(args.json_output_dir, data.graph.graph['smiles'] + '_cg_{}.json'.format(actual_num_cg))

            if os.path.exists(fpath):
                os.remove(fpath)
            with open(fpath, 'w') as f:
                json.dump(result_json, f, indent=4)

            if args.inference_method == 'graclus':
                break  # because graclus does not need num of cg beads


def main():
    args = arg_parse()
    assert args.pretrained_ckpt is not None, '--pretrained_ckpt is required.'
    assert args.json_output_dir is not None, '--json_output_dir is required.'
    args.devices = [int(device_id) for device_id in args.devices.split(',')]

    args.json_output_dir = os.path.join(args.json_output_dir, args.inference_method)

    if not os.path.exists(args.json_output_dir):
        os.makedirs(args.json_output_dir)

    test_set = HAMPerFile(data_root=args.data_root, cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat,
                          charge_feat=args.use_charge_feat, aromatic_feat=args.use_aromatic_feat, automorphism=args.automorphism)

    test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=args.num_workers,
                                     pin_memory=True)

    model = DSGPM_TP(args.input_dim, args.hidden_dim,
                  args.output_dim).cuda()
    ckpt = torch.load(args.pretrained_ckpt)
    model.load_state_dict(ckpt)

    with torch.no_grad():
        eval(test_dataloader, model, args)


if __name__ == '__main__':
    main()
