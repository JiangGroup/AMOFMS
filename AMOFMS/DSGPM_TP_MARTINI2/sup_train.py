import torch.optim as optim
import random
import tqdm
import torch.nn as nn

from option import arg_parse
from dataset.ham import HAM
from torch_geometric.loader import DataLoader, DataListLoader
from model.networks import DSGPM_TP
from model.losses import TripletLoss, PosPairMSE
from utils.util import *
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data.sampler import SubsetRandomSampler
from utils.stat import AverageMeter, FoldEpochMat
from train import eval


parent_folder = './ckpt/cgloss_10'
average_path = f'{parent_folder}/average_model_parameters.pth'
test_ratio = 0.1


def train(fold, epoch, train_loader, model, pos_pair_mse_criterion, triplet_criterion, optimizerG, args):
    model.train()
    triplet_loss_meter = AverageMeter()
    pos_pair_loss_meter = AverageMeter()
    cg_type_loss_meter = AverageMeter()

    train_loader = iter(train_loader)

    tbar = tqdm.tqdm(enumerate(train_loader), total=len(train_loader), dynamic_ncols=True)

    for i, data in tbar:
        data = data.to(torch.device(0))
        model.zero_grad()

        fg_embed, fg_cg_type_pred = model(data)

        loss = 0
        pos_pair_loss = args.pos_pair_weight * pos_pair_mse_criterion(fg_embed, data.pos_pair_index)
        loss += pos_pair_loss

        pos_pair_loss_meter.update(pos_pair_loss.item())

        if torch.numel(data.triplet_index) > 0:
            triplet_loss = args.triplet_weight * triplet_criterion(fg_embed, data.triplet_index)
            loss += triplet_loss
            triplet_loss_meter.update(triplet_loss.item())

        # compute loss of cg type matching
        criterion = nn.CrossEntropyLoss()
        cg_type_loss = args.cg_type_loss_parameter * criterion(fg_cg_type_pred, data.atom_CG_types.reshape(-1))
        cg_type_loss_meter.update(cg_type_loss.item())

        loss += cg_type_loss

        loss.backward()

        optimizerG.step()

        tbar.set_description('fold:%d [%d/%d] triplet: %.4f, pos_pair: %.4f, cg_type: %.4f'
                  % (fold+1, epoch, args.epoch, triplet_loss_meter.avg, pos_pair_loss_meter.avg, cg_type_loss_meter.avg))

    return triplet_loss_meter.avg, pos_pair_loss_meter.avg, cg_type_loss_meter.avg



def main():
    args = arg_parse()
    assert args.ckpt is not None, '--ckpt is required'
    args.devices = [int(device_id) for device_id in args.devices.split(',')]

    train_set = HAM(data_root=args.data_root, dataset_type='train', cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat,
                    charge_feat=args.use_charge_feat, aromatic_feat=args.use_aromatic_feat, cross_validation=True, automorphism=True)
    test_set = HAM(data_root=args.data_root, dataset_type='test', cycle_feat=args.use_cycle_feat, degree_feat=args.use_degree_feat,
                   charge_feat=args.use_charge_feat, aromatic_feat=args.use_aromatic_feat, cross_validation=True, automorphism=True)
    assert len(train_set) == len(test_set)

    indices = list(range(len(train_set)))
    random.shuffle(indices)

    test_set_len = int(len(train_set) * test_ratio)
    test_indices = indices[:test_set_len]
    train_indices = list(set(indices) - set(test_indices))
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_dataloader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True)
    test_dataloader = DataListLoader(test_set, batch_size=1, num_workers=0, sampler=test_sampler,
                                     pin_memory=True)

    model = DSGPM_TP(args.input_dim, args.hidden_dim,
                  args.output_dim, args=args).cuda()
    # load the pretrained dict
    pretrained_state_dict = torch.load(average_path)
    model.load_state_dict(pretrained_state_dict)


    pos_pair_mse_criterion = PosPairMSE().cuda()
    triplet_criterion = TripletLoss(args.margin).cuda()
    optimizerG = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    run_name = get_run_name(args.title)

    ckpt_dir = os.path.join(args.ckpt, run_name)
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    if args.tb_log:
        tensorboard_dir = os.path.join(args.tb_root, run_name)
        if not os.path.exists(tensorboard_dir):
            os.mkdir(tensorboard_dir)

        writer = SummaryWriter(tensorboard_dir)

    fold_epoch_matrix_manager = FoldEpochMat(args.fold, args.epoch, ['ami', 'cg_type_prec'],
                                             'ami', 'cut_prec', 'cut_recall', 'cut_fscore', 'cg_type_prec', 'cg_type_recall', 'cg_type_fscore')


    for e in range(1, args.epoch + 1):
        triplet_loss, pos_pair_loss, cg_type_loss = train(0, e, train_dataloader, model, pos_pair_mse_criterion,
                                                          triplet_criterion, optimizerG, args)

        if args.tb_log:
            writer.add_scalar('triplet loss', triplet_loss, e)
            writer.add_scalar('pos pair loss', pos_pair_loss, e)
            writer.add_scalar('cg type loss', cg_type_loss, e)

        if e % args.eval_interval == 0 and (args.start_eval_epoch is None or (e >= args.start_eval_epoch)):
            with torch.no_grad():
                test_adjusted_mutual_info, test_edge_cut_prec, test_edge_cut_recall, test_edge_cut_f_score, test_type_prec, test_type_recall, test_type_f_score = eval(0, e, test_dataloader, model, args)

            fold_epoch_matrix_manager.update(0, e - 1, {'ami': test_adjusted_mutual_info,
                                                               'cut_prec': test_edge_cut_prec,
                                                               'cut_recall': test_edge_cut_recall,
                                                               'cut_fscore': test_edge_cut_f_score,
                                                               'cg_type_prec': test_type_prec,
                                                               'cg_type_recall': test_type_recall,
                                                               'cg_type_fscore': test_type_f_score})
            best_epoch, _ = fold_epoch_matrix_manager.update_best_epoch(0)
            if best_epoch != e:
                state_dict = model.module.state_dict() if not isinstance(model, DSGPM_TP) else model.state_dict()
                torch.save(state_dict, os.path.join(ckpt_dir,
                                                    f'best_epoch.pth'))

            if not args.debug and args.tb_log:
                writer.add_scalar('test_adjusted_mutual_info', test_adjusted_mutual_info, e)
                writer.add_scalar('test_edge_cut_precision', test_edge_cut_prec, e)
                writer.add_scalar('test_edge_cut_recall', test_edge_cut_recall, e)
                writer.add_scalar('test_edge_cut_f_score', test_edge_cut_f_score, e)
                writer.add_scalar('test_type_prec', test_type_prec, e)
                writer.add_scalar('test_type_recall', test_type_recall, e)
                writer.add_scalar('test_type_fscore', test_type_f_score, e)

    best_epoch, epoch_metrics = fold_epoch_matrix_manager.update_best_epoch(0)
    print('\n[{}/{}] cross validation result:'.format(0, args.fold))
    print(f'best_epoch: {best_epoch}')
    for met, values in epoch_metrics.items():
        if met not in ['fold', 'best_epoch']:
            print(f'{met}: {values:.4f}')
    print('\n')


if __name__ == '__main__':
    main()
