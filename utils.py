import numpy as np
import math
import os
import sys

def error(msg):
    print(msg)
    sys.exit(0)


def convert_labels(name):
    if name == 'passenger_ship':
        return 0
    elif name == 'cargo_ship':
        return 1
    elif name == 'special_ship':
        return 2
    elif name == 'boat':
        return 3
    elif name == 'buoy':
        return 4
    else:
        print('convrt_to_label')
        exit()

def file_lines_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content


def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0)  # insert 0.0 at begining of list
    rec.append(1.0)  # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0)  # insert 0.0 at begining of list
    prec.append(0.0)  # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)  # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def log_average_miss_rate(precision, fp_cumsum, num_images):
    """
        log-average miss rate:
            Calculated by averaging miss rates at 9 evenly spaced FPPI points
            between 10e-2 and 10e0, in log-space.

        output:
                lamr | log-average miss rate
                mr | miss rate
                fppi | false positives per image

        references:
            [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
               State of the Art." Pattern Analysis and Machine Intelligence, IEEE
               Transactions on 34.4 (2012): 743 - 761.
    """

    if precision.size == 0:
        lamr = 0
        mr = 1
        fppi = 0
        return lamr, mr, fppi

    fppi = fp_cumsum / float(num_images)
    mr = (1 - precision)

    fppi_tmp = np.insert(fppi, 0, -1.0)
    mr_tmp = np.insert(mr, 0, 1.0)

    ref = np.logspace(-2.0, 0.0, num=9)
    for i, ref_i in enumerate(ref):
        j = np.where(fppi_tmp <= ref_i)[-1][-1]
        ref[i] = mr_tmp[j]

    lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

    return lamr, mr, fppi


def preprocess_all(gt_path, dr_path, class_names):
    gt_image_ids = os.listdir(gt_path)
    dr_image_ids = os.listdir(dr_path)

    gt_results = {}
    gt_images = []
    gt_bboxes = []

    dr_results = []

    for idx, gt_image_id in enumerate(gt_image_ids):

        # check file name, then allocate number
        g_image_id = os.path.splitext(gt_image_id)[0]

        dr_image_id = dr_image_ids[idx]
        d_image_id = os.path.splitext(dr_image_id)[0]

        if g_image_id != d_image_id:
            print('gt_image_id : %s vs dr_image_id : %s' % (g_image_id, d_image_id))
            exit()

        new_image_id = '%06d' % idx

        # ground truth -----------------------------------------------------------------------------------------------#
        gt_lines_list = file_lines_to_list(os.path.join(gt_path, gt_image_id))
        boxes_per_image = []
        image = {}
        image_id = os.path.splitext(gt_image_id)[0]
        image['file_name'] = image_id + '.jpg'
        image['width'] = 1
        image['height'] = 1
        image['id'] = idx

        for line in gt_lines_list:
            difficult = 0
            if "difficult" in line:
                line_split = line.split()
                left, top, right, bottom, _difficult = line_split[-5:]
                class_name = ""
                for name in line_split[:-5]:
                    class_name += name + " "
                class_name = class_name[:-1]
                difficult = 1
            else:
                line_split = line.split()
                left, top, right, bottom = line_split[-4:]
                class_name = ""
                for name in line_split[:-4]:
                    class_name += name + " "
                class_name = class_name[:-1]

            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            cls_id = class_names.index(class_name) + 1
            bbox = [left, top, right - left, bottom - top, difficult, int(new_image_id), cls_id,
                    (right - left) * (bottom - top)]
            boxes_per_image.append(bbox)
        gt_images.append(image)
        gt_bboxes.extend(boxes_per_image)

        # detection results ------------------------------------------------------------------------------------------#
        dr_lines_list = file_lines_to_list(os.path.join(dr_path, dr_image_id))
        #image_id = os.path.splitext(dr_image_id)[0]
        for line in dr_lines_list:
            line_split = line.split()
            confidence, left, top, right, bottom = line_split[-5:]
            class_name = ""
            for name in line_split[:-5]:
                class_name += name + " "
            class_name = class_name[:-1]
            left, top, right, bottom = float(left), float(top), float(right), float(bottom)
            result = {}
            result["image_id"] = int(new_image_id)
            result["category_id"] = class_names.index(class_name) + 1
            result["bbox"] = [left, top, right - left, bottom - top]
            result["score"] = float(confidence)
            dr_results.append(result)

    # ground truth ---------------------------------------------------------------------------------------------------#
    gt_results['images'] = gt_images
    categories = []
    for i, cls in enumerate(class_names):
        category = {}
        category['supercategory'] = cls
        category['name'] = cls
        category['id'] = i
        categories.append(category)
    gt_results['categories'] = categories

    annotations = []
    for i, box in enumerate(gt_bboxes):
        annotation = {}
        annotation['area'] = box[-1]
        annotation['category_id'] = box[-2]
        annotation['image_id'] = box[-3]
        annotation['iscrowd'] = box[-4]
        annotation['bbox'] = box[:4]
        annotation['id'] = i
        annotations.append(annotation)
    gt_results['annotations'] = annotations

    return gt_results, dr_results



