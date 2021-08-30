# -*- coding:utf-8 -*-
import mmcv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab, json
#import mmcv

if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument("-g", "--gt", type=str, help="Assign the groud true path.", default=None)
    #parser.add_argument("-d", "--dt", type=str, help="Assign the detection result path.", default=None)
    #args = parser.parse_args()
    labels = mmcv.load('labels.json')
    annFile = 'json_files/instances_val2017.json'
    cocoGt = COCO(annFile)

    resFile = 'result.json'
    cocoDt = cocoGt.loadRes(resFile)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
