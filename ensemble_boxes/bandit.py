import mmcv
import numpy as np
import mmcv
from collections import defaultdict
import math

bandit_theta = 0.5

def bandit_score_transform(dt_results, gt_results):
    '''
    :param dt_result: predictions
    :param gt_result: annotations
    :return: score transform list
    '''

    ########## read detections ##########
    results_dict = defaultdict(list)
    all_box_num = len(dt_results)

    gt_dict = dict()
    for tmp in gt_results['annotations']:
        if str(tmp['image_id']) not in gt_dict:
            gt_dict[str(tmp['image_id'])] = dict()

    for tmp in gt_results['annotations']:
        if str(tmp['category_id']) not in gt_dict[str(tmp['image_id'])]:
            gt_dict[str(tmp['image_id'])][str(tmp['category_id'])] = [tmp['bbox']]
        else:
            gt_dict[str(tmp['image_id'])][str(tmp['category_id'])].append(tmp['bbox'])


    ##### compute SP at different conf levels #####
    score_list = [0.04 * (i + 1) for i in range(25)]
    tp_list = []
    total_box = []
    error_list = []
    for i in range(len(score_list) - 1):
        # print(i)
        min_score, max_score = score_list[i], score_list[i + 1]
        dt_dict = dict()
        for tmp in dt_results:
            if str(tmp['image_id']) not in dt_dict:
                dt_dict[str(tmp['image_id'])] = dict()
        # print(len(gt_dict.keys()), len(dt_dict.keys()))
        # input()
        total_bbox_num = 1
        for tmp in dt_results:
            if (tmp['score'] < min_score) or (tmp['score'] > max_score):
                continue
            elif str(tmp['category_id']) not in dt_dict[str(tmp['image_id'])]:
                value = tmp['bbox']  # .append(0)
                dt_dict[str(tmp['image_id'])][str(tmp['category_id'])] = [value]
            else:
                value = tmp['bbox']  # .append(0)
                dt_dict[str(tmp['image_id'])][str(tmp['category_id'])].append(value)
            total_bbox_num += 1

        tp = 0
        for cur_id in dt_dict:
            if cur_id not in gt_dict:
                continue

            for cur_cls in dt_dict[cur_id]:
                if cur_id not in gt_dict:
                    continue
                elif cur_cls not in gt_dict[cur_id]:
                    continue
                else:
                    gt_box = np.array(gt_dict[cur_id][cur_cls])
                    dt_box = np.array(dt_dict[cur_id][cur_cls])
                    for tmp_dt_box in dt_box:
                        cur_x1 = tmp_dt_box[0]
                        cur_y1 = tmp_dt_box[1]
                        cur_x2 = tmp_dt_box[0] + tmp_dt_box[2]
                        cur_y2 = tmp_dt_box[1] + tmp_dt_box[3]

                        x1 = gt_box[:, 0]
                        y1 = gt_box[:, 1]
                        x2 = gt_box[:, 0] + gt_box[:, 2]
                        y2 = gt_box[:, 1] + gt_box[:, 3]

                        dt_areas = tmp_dt_box[2] * tmp_dt_box[3]
                        gt_areas = (x2 - x1) * (y2 - y1)

                        xx1 = np.maximum(cur_x1, x1)
                        yy1 = np.maximum(cur_y1, y1)
                        xx2 = np.minimum(cur_x2, x2)
                        yy2 = np.minimum(cur_y2, y2)

                        w = np.maximum(0.0, xx2 - xx1)
                        h = np.maximum(0.0, yy2 - yy1)
                        inter = w * h
                        ovr = inter / (dt_areas + gt_areas - inter)
                        merge_inds = np.where(ovr > 0.5)
                        if len(merge_inds[0]) > 0:
                            tp += 1
        ucb_value = tp / total_bbox_num + bandit_theta * math.sqrt(2 * math.log(all_box_num) / total_bbox_num)

        total_box.append(total_bbox_num)
        tp_list.append(ucb_value)

    a = score_list[1:]
    for i, tmp in enumerate(a):
        a[i] = tmp - 0.02
    value = [a, tp_list]

    return value



    