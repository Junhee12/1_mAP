from mean_average_precision import MetricBuilder

import glob

from utils import *

metric_fn = MetricBuilder.build_evaluation_metric("map_2d", async_mode=True, num_classes=5)

GT_PATH = 'GT'
DR_PATH = 'DR'

ground_truth_files_list = glob.glob(GT_PATH + '/*.txt')
ground_truth_files_list.sort()

dr_files_list = glob.glob(DR_PATH + '/*.txt')
dr_files_list.sort()

# 정답지에서 박스 목록 추출
for txt_file in ground_truth_files_list:
    file_id = txt_file.split(".txt", 1)[0]
    file_id = os.path.basename(os.path.normpath(file_id))
    temp_path = os.path.join(DR_PATH, (file_id + ".txt"))
    lines_list = file_lines_to_list(txt_file)

    gt_record_list = []
    for line in lines_list:

        class_name, left, top, right, bottom = line.split()

        # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
        record = [int(left), int(top), int(right), int(bottom), convert_labels(class_name), 0, 0]
        gt_record_list.append(record)

    gt = np.array(gt_record_list)

    if not os.path.exists(temp_path):
        error_msg = "Error. File not found: {}\n".format(temp_path)
        error(error_msg)

    dr_record_list = []
    dr_lines_list = file_lines_to_list(temp_path)
    for line in dr_lines_list:

        class_name, conf, left, top, right, bottom = line.split()

        # [xmin, ymin, xmax, ymax, class_id, confidence]]
        record = [int(left), int(top), int(right), int(bottom), convert_labels(class_name), float(conf)]
        dr_record_list.append(record)

    dr = np.array(dr_record_list)

    # preds: [xmin, ymin, xmax, ymax, class_id, confidence]
    # gt: [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    metric_fn.add(dr, gt)


# iou_thresholds (list of float): IOU thresholds.
# recall_thresholds (np.array or None): specific recall thresholds to the computation of average precision.
# mpolicy (str): box matching policy. greedy - greedy matching like VOC PASCAL. soft - soft matching like COCO.
metric = metric_fn.value(iou_thresholds=0.5, recall_thresholds=np.arange(0., 1.1, 0.1), mpolicy='greedy')
#print(metric)
print(f"VOC PASCAL mAP: {metric['mAP']*100}")

# compute PASCAL VOC metric at the all points
metric = metric_fn.value(iou_thresholds=0.5,  mpolicy='greedy')
print(f"VOC PASCAL mAP in all points: {metric['mAP']*100}")

# compute metric COCO metric
metric = metric_fn.value(iou_thresholds=np.arange(0.5, 1.0, 0.05), recall_thresholds=np.arange(0., 1.01, 0.01), mpolicy='soft')
print(f"COCO mAP: {metric['mAP']*100}")
