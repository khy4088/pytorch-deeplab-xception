import argparse
import os
import numpy as np 
import tqdm
import torch
from pycocotools.coco import COCO



from PIL import Image
from dataloaders import make_data_loader
from modeling.deeplab import *
from dataloaders.utils import get_pascal_labels
from utils.metrics import Evaluator



class Tester(object):
    def __init__(self, args):
        if not os.path.isfile(args.model):
            raise RuntimeError("no checkpoint found at '{}'".fromat(args.model))
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.args = args
        self.color_map = get_pascal_labels()
        self.train_loader, self.val_loader, self.test_loader, self.train_ids, self.val_ids, self.test_ids, self.nclass = make_data_loader(args)
        self.testannFile = COCO('/content/dlv3_dataset/annotations/instances_test.json')
        self.test_ids = self.testannFile.getImgIds()
        #Define model
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=False,
                        freeze_bn=False)
        self.model = model
        checkpoint = torch.load(args.model, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        model = model.to(self.device)
        print('loaded model from {}'.format(args.model))

        self.evaluator = Evaluator(self.nclass)

    def save_image(self, imgId, array, op):
        text = 'gt'
        if op == 0:
            text = 'pred'
        file_name = str(imgId)+'_'+text+'.png'
        r = array.copy()
        g = array.copy()
        b = array.copy()

        for i in range(self.nclass):
            r[array == i] = self.color_map[i][0]
            g[array == i] = self.color_map[i][1]
            b[array == i] = self.color_map[i][2]
    
        rgb = np.dstack((r, g, b))

        save_img = Image.fromarray(rgb.astype('uint8'))
        save_img.save(self.args.save_path+os.sep+file_name)
        print(self.args.save_path+os.sep+file_name, "  saved")

    def test(self):
        self.model.eval()
        self.evaluator.reset()
        # tbar = tqdm(self.test_loader, desc='\r')
        for i, sample in enumerate(self.test_loader):
            image, target = sample['image'], sample['label']
            imgname = sample['file_name']
            imgname = ''.join(imgname).rstrip('.jpg')
            print(imgname, type(imgname))
            image = image.to(self.device)
            target = target.to(self.device)
            with torch.no_grad():
                output = self.model(image)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            self.save_image(imgname, pred[0], 0)
            self.save_image(imgname, target[0], 1)
            self.evaluator.add_batch(target, pred)
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        print('Acc:{}, Acc_class:{}'.format(Acc, Acc_class))
        print('mIOU:{}, FWIoU:{}'.format(mIoU, FWIoU))


def main():
    parser = argparse.ArgumentParser(description='Pytorch DeeplabV3Plus Test your data')

    parser.add_argument('--test', action='store_true', default=True, 
                        help='test your data')
    parser.add_argument('--dataset', default='coco', 
                        help='datset format')
    parser.add_argument('--backbone', default='resnet', 
                        help='what is your network backbone')
    parser.add_argument('--out_stride', type=int, default=16,
                        help='output stride')
    parser.add_argument('--crop_size', type=int, default=1080,
                        help='image size')
    parser.add_argument('--model', type=str, default=None,
                        help='load your model')
    parser.add_argument('--save_path', type=str, default=None,
                        help='save your prediction data')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='batch size')

    args = parser.parse_args()
    
    if args.test:
        tester = Tester(args)
        tester.test()

if __name__ == "__main__":
    main()