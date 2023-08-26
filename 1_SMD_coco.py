from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import os
import json

from utils import *

GT_PATH = 'GT'
DR_PATH = 'DR'

class_names = ['passenger_ship', 'cargo_ship', 'special_ship', 'boat', 'buoy']

COCO_PATH = os.path.join('1_coco_eval')

if not os.path.exists(COCO_PATH):
    os.makedirs(COCO_PATH)

GT_JSON_PATH = os.path.join(COCO_PATH, 'instances_gt.json')
DR_JSON_PATH = os.path.join(COCO_PATH, 'instances_dr.json')

results_gt, results_dr = preprocess_all(GT_PATH, DR_PATH, class_names)

with open(GT_JSON_PATH, "w") as f:
    json.dump(results_gt, f, indent=4)

with open(DR_JSON_PATH, "w") as f:
    json.dump(results_dr, f, indent=4)

cocoGt = COCO(GT_JSON_PATH)
cocoDt = cocoGt.loadRes(DR_JSON_PATH)
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()