import sys
import random
import os
import re
from rdkit.Chem import AllChem
import requests
import tqdm
import json
import copy
from model.networks import DSGPM_TP
from model.graph_cuts import graph_cuts
from utils.post_processing import enforce_connectivity
import torch.nn.functional as F
from dataset.ham import ATOMS
from sklearn.exceptions import UndefinedMetricWarning
from collections import Counter
import torch
import seaborn as sns
import numpy as np
import io
from dataset.ham_per_file import HAMPerFile
from torch_geometric.data import DataListLoader
from PIL import Image
from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from skimage.io import imsave
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UndefinedMetricWarning)
simplefilter(action='ignore', category=Warning)

import argparse

svg = False
debug = False

def draw_graph(graph, hard_assign, cg_types):

    smiles = graph.graph['smiles']
    molecule = Chem.MolFromSmiles(smiles)

    assert molecule is not None
    rdDepictor.Compute2DCoords(molecule)

    palette = np.array(sns.hls_palette(hard_assign.max() + 1))

    atom_index = list(range(len(graph.nodes)))
    undirected_edges = np.array([(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in molecule.GetBonds()])
    non_cut_edges_indices = np.nonzero(hard_assign[undirected_edges[:, 0]] == hard_assign[undirected_edges[:, 1]])[0]
    bond_colors = palette[hard_assign[undirected_edges[non_cut_edges_indices][:, 0]]]
    bond_colors = list(map(tuple, bond_colors))
    atom_colors = list(map(tuple, palette[hard_assign]))

    atom_id_to_color_dict = dict(zip(atom_index, atom_colors))
    non_edge_idx_to_color_dict = dict(zip(non_cut_edges_indices.tolist(), bond_colors))

    if svg:
        drawer = rdMolDraw2D.MolDraw2DSVG(1200, 600)
    else:
        drawer = rdMolDraw2D.MolDraw2DCairo(1200, 600)
    for l in range(molecule.GetNumAtoms()):
        molecule.GetAtomWithIdx(l).SetProp('atomNote', cg_types[l])
    drawer.drawOptions().addAtomIndices = True
    drawer.DrawMolecule(
        molecule,
        highlightAtoms=atom_index,
        highlightBonds=non_cut_edges_indices.tolist(),
        highlightAtomColors=atom_id_to_color_dict,
        highlightBondColors=non_edge_idx_to_color_dict,
        highlightAtomRadii=dict(zip(atom_index, [0.1] * len(atom_index)))
    )

    drawer.FinishDrawing()
    if svg:
        img = drawer.GetDrawingText().replace('svg:','')
        #================write to files============================
    else:
        txt = drawer.GetDrawingText()
        img = Image.open(io.BytesIO(txt))
        img = np.asarray(img)

    return img


def gen_vis(dataloader, output_file):
    vis_path = output_file


    for i, data in enumerate(dataloader):
        # skip saved smiles
        data = data[0]
        num_nodes = data.x.shape[0]
        data.batch = torch.zeros(num_nodes).long()
        graph_nx = data.graph
        cg_types = data.json['cgnode_types']
        gt_hard_assigns = data.y.cpu().numpy()

        if not debug:
            gt_img = draw_graph(graph_nx, gt_hard_assigns, cg_types)
            print('success: {}'.format(graph_nx.graph['smiles']))

            if svg:
                fpath = os.path.join(vis_path, data.fname + '.svg')
                svg_file = open(fpath, "wt")
                svg_file.write(gt_img)
                svg_file.close()
            else:
                fpath = os.path.join(vis_path, graph_nx.graph['smiles'] + '.png')
                imsave(fpath, gt_img)



def adjust_list(lst):
    counts = Counter(lst)
    max_count = counts.most_common(1)[0][1]
    most_common_elements = [element for element, count in counts.items() if count == max_count]
    chosen_element = random.choice(most_common_elements)
    return [chosen_element] * len(lst)


CG_TYPE_DICT = {'C1': 0, 'C2': 1, 'C3': 2, 'C4': 3, 'C5': 4, 'NDA': 5, 'ND': 6, 'NA': 7, 'N0': 8, 'P1': 9, 'P2': 10,
                'P3': 11, 'P4': 12, 'P5': 13, 'QDA': 14, 'QD': 15, 'QA': 16, 'Q0': 17}
CG_TYPE_DICT = {value: key for key, value in CG_TYPE_DICT.items()}

def eval(test_dataloader, model, output_dir, num_cg_beads=None, use_regular_mapping_from_prediction=True, cluster_random_seed=None):
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

        if num_cg_beads is None:
            iter_num_cg_beads = range(2, num_nodes)
        else:
            iter_num_cg_beads = num_cg_beads


        hard_assign, _ = graph_cuts(fg_embed, data.edge_index, num_cg_beads, random_state=cluster_random_seed)
        # print(hard_assign)
        hard_assign = enforce_connectivity(hard_assign, edge_index_cpu)
        # print(hard_assign)
        actual_num_cg = max(hard_assign) + 1
        if actual_num_cg != num_cg_beads:
            print('warning: actual vs. expected: {} vs. {}'.format(actual_num_cg, num_cg_beads))

        result_json = copy.deepcopy(json_data)
        for atom_idx, cg_idx in enumerate(hard_assign):
            result_json['nodes'][atom_idx]['cg_id'] = int(cg_idx)
            result_json['nodes'][atom_idx]['cg_type'] = predicted_cg_types[atom_idx]
        result_json['cgnode_types'] = predicted_cg_types

        for cg_idx in range(num_cg_beads):
            atom_indices = np.nonzero(hard_assign == cg_idx)[0].tolist()
            atom_indices = [int(x) for x in atom_indices]
            result_json['cgnodes'].append(atom_indices)

        if use_regular_mapping_from_prediction:
            cg_groups = []
            for i in range(num_cg_beads):
                cg_groups_tmp = []
                for id, cg in enumerate(hard_assign):
                    if i == cg:
                        cg_groups_tmp.append(predicted_cg_types[id])
                cg_groups.append(cg_groups_tmp)

            new_cg_groups = [adjust_list(sublist) for sublist in cg_groups]


            new_predicted_cg_types = [new_cg_groups[i][0] for i in hard_assign]
            for atom_idx, cg_idx in enumerate(hard_assign):
                result_json['nodes'][atom_idx]['cg_type'] = new_predicted_cg_types[atom_idx]
            result_json['cgnode_types'] = new_predicted_cg_types

        fpath = os.path.join(output_dir, data.graph.graph['smiles'] + '_cg_{}.json'.format(actual_num_cg))

        if os.path.exists(fpath):
            os.remove(fpath)
        with open(fpath, 'w') as f:
            json.dump(result_json, f, indent=4)

        return result_json


CACTUS = "https://cactus.nci.nih.gov/chemical/structure/{0}/{1}"
commad_file = './command.log'


def smiles_to_iupac(smiles):
    rep = "iupac_name"
    url = CACTUS.format(smiles, rep)
    response = requests.get(url)
    response.raise_for_status()
    return response.text


def DSGPM_TPtoCG(smiles, file_dir, num_cg_bead, name=None, cluster_random_seed=None):
    '''Takes in a PDB file or a SMILES string and returns one bead mapping
       in JSON format and .png based on the DGSPM model prediction.

       file_dir : path to the output
       smile : SMILES string
       num_cg_bead: the number of cg bead you want to predict
    '''

    m = Chem.MolFromSmiles(smiles)
    edges = []
    for j in range(m.GetNumBonds()):
        begin = m.GetBonds()[j].GetBeginAtomIdx()
        end = m.GetBonds()[j].GetEndAtomIdx()
        bond = m.GetBondWithIdx(j).GetBondTypeAsDouble()
        value = {"source": begin, "target": end, "bondtype": bond}
        edges.append(value)

    # Create one bead mappings
    nodes = []
    cgnodes = []
    cgnode_types = []

    for l in range(m.GetNumAtoms()):
        element = m.GetAtomWithIdx(l).GetSymbol()
        val = {"id": l, "element": element, "charge": 0, "cg_id": 0, "cg_type": 0}
        nodes.append(val)

    AllChem.ComputeGasteigerCharges(m)
    t_charge = 0
    for l in range(m.GetNumAtoms()):
        nodes[l]["charge"] = round(float(m.GetAtomWithIdx(l).GetProp('_GasteigerCharge')), 3)
        t_charge += nodes[l]["charge"]

    try:
        smiles_name = smiles_to_iupac(smiles)
    except:
        print("\nThe IUPAC name of commpound is not found! ")
        smiles_name = None

    # Create a nested dictionary to be given in json format
    cg_dict = {"compound name": smiles_name, "smiles": smiles, "cgnodes": cgnodes,
               "cgnode_types": cgnode_types, "nodes": nodes, "edges": edges, "note": "generated by Drep Zhong script"}

    # Writing to json file

    out_dir = file_dir

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    if name is None:
        name = smiles

    oname = re.sub('[^A-Za-z0-9]+', '', name) + '.json'
    ofile = os.path.join(out_dir, oname)
    with open(ofile, 'w') as f:
        f.write(json.dumps(cg_dict, sort_keys=False, indent=4))

    test_set = HAMPerFile(data_root=out_dir, cycle_feat=True, degree_feat=True,
                          charge_feat=True, aromatic_feat=False,
                          automorphism=False)

    test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=4,
                                     pin_memory=True)

    model = DSGPM_TP(input_dim=len(ATOMS), hidden_dim=128,
                  embedding_dim=128).cuda()
    ckpt = torch.load(os.path.join(os.path.dirname(__file__), 'model/best_epoch.pth'))
    model.load_state_dict(ckpt)

    predict_out_dir = os.path.join(file_dir, 'CG')
    if not os.path.exists(predict_out_dir):
        os.mkdir(predict_out_dir)
    with torch.no_grad():
        predict_json = eval(test_dataloader, model, output_dir=predict_out_dir, num_cg_beads=num_cg_bead,
                            cluster_random_seed=cluster_random_seed)

    test_set = HAMPerFile(data_root=predict_out_dir, cycle_feat=True, degree_feat=True,
                          charge_feat=True, aromatic_feat=True,
                          automorphism=False)
    test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=0, pin_memory=True)
    gen_vis(test_dataloader, output_file=predict_out_dir)

    print('DSGPM_TP prediction complete.')
    return predict_json


def main():
    # pass
    # mol_form = 'sml'
    # smile = 'CCCCCCCCCCCCCCCC'
    # pdb_file = False
    # json_output = './'
    # if not mol_form == 'sml':
    #     smile = Chem.MolToSmiles(AllChem.MolFromPDBFile(pdb_file))
    #
    # num_bead = 4
    # DSGPM_TPtoCG(smile=smile, file_dir='debug_test', num_cg_bead=num_bead)

    parser = argparse.ArgumentParser(description="========= DSGPM-TP model for MARTINI2 CG Mapping =========")
    parser.add_argument('--mol_form', type=str, default='sml', help='Molecular format (default: sml, smiles)')
    parser.add_argument('--smiles', type=str, default='CCCCCCCCCCCCCCCC', help='SMILES string of the molecule')
    parser.add_argument('--pdb_file', type=str, default=None, help='Path to the PDB file')
    parser.add_argument('--json_output', type=str, default='./', help='Path to the JSON output directory')
    parser.add_argument('--num_bead', type=int, default=4, help='Number of CG beads (default: 4)')

    args = parser.parse_args()

    if args.mol_form != 'sml' and args.pdb_file:
        smiles = Chem.MolToSmiles(AllChem.MolFromPDBFile(args.pdb_file))
    else:
        smiles = args.smiles

    if not os.path.exists(args.json_output):
        os.makedirs(args.json_output)

    DSGPM_TPtoCG(smiles=smiles, file_dir=args.json_output, num_cg_bead=args.num_bead)

if __name__ == "__main__":
    main()