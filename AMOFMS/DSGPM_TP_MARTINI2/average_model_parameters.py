from option import arg_parse
from model.networks import DSGPM_TP
from utils.util import average_model_parameters


args = arg_parse()
args.devices = [int(device_id) for device_id in args.devices.split(',')]
train_model = DSGPM_TP(args.input_dim, args.hidden_dim, args.output_dim, args=args).cuda()

parent_folder = './ckpt/cgloss_10'
average_path = f'{parent_folder}/average_model_parameters.pth'
fold_pth = 'best_epoch.pth'

average_model_parameters(parent_folder=parent_folder, train_model=train_model, pth_name=fold_pth, save_path=average_path)