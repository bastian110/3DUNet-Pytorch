from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import config
from utils import logger, common
from dataset.dataset_lits_test import Test_Datasets, to_one_hot_3d
import SimpleITK as sitk
import os
import numpy as np
from models import ResUNet, UNet, KiUNet, SegNet
from utils.metrics import DiceAverage, pixel_acc, recall, precision, specificity
from collections import OrderedDict


def predict_one_img(model, img_dataset, args):
    dataloader = DataLoader(dataset=img_dataset, batch_size=1, num_workers=0, shuffle=False)
    model.eval()
    test_dice = DiceAverage(args.n_labels)
    test_pixel_accuracy = pixel_acc(args.n_labels)
    test_precision = precision(args.n_labels)
    test_specificity = specificity(args.n_labels)
    test_recall = recall(args.n_labels)
    target = to_one_hot_3d(img_dataset.label, args.n_labels)

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            data = data.to(device)
            output = model(data)
            # output = nn.functional.interpolate(output, scale_factor=(1//args.slice_down_scale,1//args.xy_down_scale,1//args.xy_down_scale), mode='trilinear', align_corners=False) # 空间分辨率恢复到原始size: La résolution spatiale est restaurée à la taille d'origine
            img_dataset.update_result(output.detach().cpu())

    pred = img_dataset.recompone_result()
    pred = torch.argmax(pred, dim=1)

    pred_img = common.to_one_hot_3d(pred, args.n_labels)
    test_dice.update(pred_img, target)
    test_pixel_accuracy.update(pred_img, target)
    test_precision.update(pred_img, target)
    test_specificity.update(pred_img, target)
    test_recall.update(pred_img, target)

    test_dice = OrderedDict({'Dice_liver': test_dice.avg[1], 'test_pixel_accuracy': test_pixel_accuracy.avg[1],
                             'test_precision': test_precision.avg[1], 'test_specificity:': test_specificity.avg[1],
                             'test_recall': test_recall.avg[1]})
    if args.n_labels == 3: test_dice.update({'Dice_tumor': test_dice.avg[2]})

    pred = np.asarray(pred.numpy(), dtype='uint8')
    if args.postprocess:
        pass  # TO DO
    pred = sitk.GetImageFromArray(np.squeeze(pred, axis=0))

    return test_dice, pred


if __name__ == '__main__':
    args = config.args
    save_path = os.path.basename('/content/drive/MyDrive/3DUNet/liver_segmentation/UNet/')
    device = torch.device('cpu' if args.cpu else 'cuda')
    # model info
    model = UNet(in_channel=1, out_channel=args.n_labels, training=False).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    ckpt = torch.load('/content/drive/MyDrive/3DUNet/liver_segmentation/UNet/best_model.pth'.format(save_path))
    model.load_state_dict(ckpt['net'])

    test_log = logger.Test_Logger(save_path, "test_log")
    # data info
    result_save_path = '/content/training/'.format(save_path)
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)

    datasets = Test_Datasets(args.test_data_path, args=args)
    for img_dataset, file_idx in datasets:
        test_dice, pred_img = predict_one_img(model, img_dataset, args)
        test_log.update(file_idx, test_dice)
        sitk.WriteImage(pred_img, os.path.join(result_save_path, 'result-' + file_idx + '.gz'))
