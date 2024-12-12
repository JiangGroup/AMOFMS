import numpy as np


class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class StdAverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.val_lst = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.val_lst.append(val)

    def std(self):
        return np.std(np.array(self.val_lst))


class FoldEpochMat:
    def __init__(self, num_fold, num_epoch, best_result_keys, *keys):
        self.best_result_keys = best_result_keys
        self.best_epoch_of_fold = {}
        self.average_all_best_epoch = {}
        self.key_to_mat_dict = {}
        self.keys = keys
        self.num_fold = num_fold
        for k in keys:
            self.key_to_mat_dict[k] = np.zeros((num_fold, num_epoch))

    def update(self, fold, epoch, key_val_dict: dict):
        for key, val in key_val_dict.items():
            self.key_to_mat_dict[key][fold, epoch] = val

    def result(self, fold):
        fold += 1
        criterion = sum([self.key_to_mat_dict[key][:fold].mean(axis=0) for key in self.best_result_keys])
        best_epoch = criterion.argmax()

        mean_ret = {}
        std_ret = {}
        for key, mat in self.key_to_mat_dict.items():
            colomn = mat[:fold, best_epoch]
            mean_ret[key] = colomn.mean()
            std_ret[key] = colomn.std()

        return mean_ret, std_ret, best_epoch + 1

    def update_best_epoch(self, fold):
        criterion = sum([self.key_to_mat_dict[key][fold:fold+1] for key in self.best_result_keys])
        best_epoch = criterion.argmax()
        ret = {}
        ret['fold'] = fold
        ret['best_epoch'] = best_epoch
        for key, mat in self.key_to_mat_dict.items():
            ret[key] = mat[fold:fold+1, best_epoch][0]

        self.best_epoch_of_fold.update({f'fold_{fold}': ret})

        return best_epoch+1, ret

    def get_best_epoch(self):
        return self.best_epoch_of_fold

    def average_all_best_epoch_of_fold(self):
        for k in self.keys:
            tmp = 0
            for _, values in self.best_epoch_of_fold.items():
                tmp += values[k]
            self.average_all_best_epoch.update({k: round(tmp/self.num_fold, 4)})
        return self.average_all_best_epoch




