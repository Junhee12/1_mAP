# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from torch import tensor
from torchmetrics.detection import MeanAveragePrecision

import glob
import os
import torch

from utils import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    GT_PATH = 'GT'
    DR_PATH = 'DR'

    ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
    ground_truth_files_list.sort()

    dr_files_list = glob.glob(DR_PATH + '/*.txt')
    dr_files_list.sort()

    # 101 spaces
    metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", class_metrics=True)

    # self.iou_thresholds
    # self.rec_thresholds

    # 정답지에서 박스 목록 추출
    for txt_file in ground_truth_files_list:

        file_id = txt_file.split(".txt", 1)[0]
        file_id = os.path.basename(os.path.normpath(file_id))
        lines_list = file_lines_to_list(txt_file)

        box_list = []
        label_list = []
        for line in lines_list:
            class_name, left, top, right, bottom = line.split()

            box = tensor([[int(left), int(top), int(right), int(bottom)]])
            label = tensor([convert_labels(class_name)])

            box_list.append(box)
            label_list.append(label)

        if len(box_list) == 0:
            print(file_id)
        else:
            target = [dict(boxes=torch.cat(box_list, 0), labels=torch.cat(label_list, 0), )]

        # -------------------------------------------------------------------------------------------------------------#
        temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
        box_list = []
        label_list = []
        conf_list = []
        dr_lines_list = file_lines_to_list(temp_path)
        for line in dr_lines_list:
            class_name, conf, left, top, right, bottom = line.split()

            box = tensor([[int(left), int(top), int(right), int(bottom)]])
            label = tensor([convert_labels(class_name)])
            conf = tensor([float(conf)])

            box_list.append(box)
            label_list.append(label)
            conf_list.append(conf)

        # [xmin, ymin, xmax, ymax, class_id, confidence]]
        preds = [dict(boxes=torch.cat(box_list, 0), scores=torch.cat(conf_list, 0), labels=torch.cat(label_list, 0))]

        metric.update(preds, target)

    from pprint import pprint

    pprint(metric.compute())
