import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class LossAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

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
        self.avg = round(self.sum / self.count, 4)
        # print(self.val)


class DiceAverage(object):
    """Computes and stores the average and current value for calculate average loss"""

    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = DiceAverage.get_dices(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_dices(logits, targets):
        dices = []
        for class_index in range(targets.size()[1]):
            inter = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            union = torch.sum(logits[:, class_index, :, :, :]) + torch.sum(targets[:, class_index, :, :, :])
            dice = (2. * inter + 1) / (union + 1)

            dices.append(dice.item())
        return np.asarray(dices)


class pixel_acc(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = pixel_acc.get_pixel_acc(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_pixel_acc(logits, targets):
        pixel = []
        for class_index in range(targets.size()[1]):
            tp = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            fp = torch.sum(logits[:, class_index, :, :, :] * (1 - targets[:, class_index, :, :, :]))
            fn = torch.sum((1 - logits[:, class_index, :, :, :]) * targets[:, class_index, :, :, :])
            tn = torch.sum((1 - logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))

            pixel_acc = (tp + tn) / (tp + tn + fp + fn)
            pixel.append(pixel_acc.item())
        return np.asarray(pixel)

        # specificity, recall


class precision(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = precision.get_precision(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_precision(logits, targets):
        prec = []
        for class_index in range(targets.size()[1]):
            tp = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            fp = torch.sum(logits[:, class_index, :, :, :] * (1 - targets[:, class_index, :, :, :]))
            fn = torch.sum((1 - logits[:, class_index, :, :, :]) * targets[:, class_index, :, :, :])
            tn = torch.sum((1 - logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))

            precision = (tp) / (tp + fp)

            prec.append(precision.item())
        return np.asarray(prec)


class specificity(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = specificity.get_specificity(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_specificity(logits, targets):
        spec = []
        for class_index in range(targets.size()[1]):
            tp = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            fp = torch.sum(logits[:, class_index, :, :, :] * (1 - targets[:, class_index, :, :, :]))
            fn = torch.sum((1 - logits[:, class_index, :, :, :]) * targets[:, class_index, :, :, :])
            tn = torch.sum((1 - logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))

            specificity = (tn) / (tn + fp)

            spec.append(specificity.item())
        return np.asarray(spec)


class recall(object):
    def __init__(self, class_num):
        self.class_num = class_num
        self.reset()

    def reset(self):
        self.value = np.asarray([0] * self.class_num, dtype='float64')
        self.avg = np.asarray([0] * self.class_num, dtype='float64')
        self.sum = np.asarray([0] * self.class_num, dtype='float64')
        self.count = 0

    def update(self, logits, targets):
        self.value = recall.get_recall(logits, targets)
        self.sum += self.value
        self.count += 1
        self.avg = np.around(self.sum / self.count, 4)
        # print(self.value)

    @staticmethod
    def get_recall(logits, targets):
        rec = []
        for class_index in range(targets.size()[1]):
            tp = torch.sum(logits[:, class_index, :, :, :] * targets[:, class_index, :, :, :])
            fp = torch.sum(logits[:, class_index, :, :, :] * (1 - targets[:, class_index, :, :, :]))
            fn = torch.sum((1 - logits[:, class_index, :, :, :]) * targets[:, class_index, :, :, :])
            tn = torch.sum((1 - logits[:, class_index, :, :, :]) * (1 - targets[:, class_index, :, :, :]))

            recall = (tp) / (tp + fn)

            rec.append(recall.item())
        return np.asarray(rec)

