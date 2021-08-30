from ensemble_boxes import *
import mmcv


iou_thr = 0.7
skip_box_thr = 0.0001
weights = [1, 1]

def ensemble_results(dt_result_1, dt_result_2, gt_result):
    '''
    :param dt_result_1: predictions_1
    :param dt_result_2: predictions_2
    :param gt_result: annotations
    :return: ensemble results
    '''
    ensemble_results = []

    dt_dict_1 = dict()
    dt_dict_2 = dict()
    gt_dict = dict()
    for tmp in gt_result['images']:
        dt_dict_1[str(tmp['id'])] = []
        dt_dict_2[str(tmp['id'])] = []
        gt_dict[str(tmp['id'])] = [tmp['width'], tmp['height']]
    for tmp in dt_result_1:
        dt_dict_1[str(tmp['image_id'])].append(tmp)
    for tmp in dt_result_2:
        dt_dict_2[str(tmp['image_id'])].append(tmp)

    k = 0
    for tmp_id in gt_dict.keys():
        k += 1
        print('id:', k)
        width, height = int(gt_dict[tmp_id][0]), int(gt_dict[tmp_id][1])
        bbox_list_1, bbox_list_2 = [], []
        score_list_1, score_list_2 = [], []
        cls_list_1, cls_list_2 = [], []
        for tmp_bbox in dt_dict_1[tmp_id]:
            t_bbox = tmp_bbox['bbox']
            t_bbox[2] = t_bbox[0] + t_bbox[2]
            t_bbox[3] = t_bbox[1] + t_bbox[3]
            t_bbox[0] = t_bbox[0] / width
            t_bbox[2] = min(t_bbox[2] / width, 1)
            t_bbox[1] = t_bbox[1] / height
            t_bbox[3] = min(t_bbox[3] / height, 1)
            bbox_list_1.append(t_bbox)
            score_list_1.append(tmp_bbox['score'])
            cls_list_1.append(tmp_bbox['category_id'])
        for tmp_bbox in dt_dict_2[tmp_id]:
            t_bbox = tmp_bbox['bbox']
            t_bbox[2] = t_bbox[0] + t_bbox[2]
            t_bbox[3] = t_bbox[1] + t_bbox[3]

            t_bbox[0] = t_bbox[0] / width
            t_bbox[2] = min(t_bbox[2] / width, 1)
            t_bbox[1] = t_bbox[1] / height
            t_bbox[3] = min(t_bbox[3] / height, 1)

            bbox_list_2.append(t_bbox)
            score_list_2.append(tmp_bbox['score'])
            cls_list_2.append(tmp_bbox['category_id'])

        if (len(bbox_list_1) == 0) and (len(bbox_list_2) > 0):
            bbox_list = [bbox_list_2]
            score_list = [score_list_2]
            cls_list = [cls_list_2]
        elif (len(bbox_list_1) > 0) and (len(bbox_list_2) == 0):
            bbox_list = [bbox_list_1]
            score_list = [score_list_1]
            cls_list = [cls_list_1]
        elif (len(bbox_list_1) == 0) and (len(bbox_list_2) == 0):
            continue
        else:
            bbox_list = [bbox_list_1, bbox_list_2]
            score_list = [score_list_1, score_list_2]
            cls_list = [cls_list_1, cls_list_2]
        new_boxes, new_scores, new_labels = p_nms(bbox_list, score_list, cls_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)

        new_boxes = new_boxes.tolist()
        new_scores = new_scores.tolist()
        new_labels = new_labels.tolist()

        for cur_bbox, cur_score, cur_cls in zip(new_boxes, new_scores, new_labels):
            cur_bbox[0] = cur_bbox[0] * width
            cur_bbox[2] = cur_bbox[2] * width
            cur_bbox[1] = cur_bbox[1] * height
            cur_bbox[3] = cur_bbox[3] * height

            cur_bbox[2] = cur_bbox[2] - cur_bbox[0]
            cur_bbox[3] = cur_bbox[3] - cur_bbox[1]

            cur_result = dict()
            cur_result['bbox'] = cur_bbox
            cur_result['score'] = cur_score
            cur_result['category_id'] = int(cur_cls)
            cur_result['image_id'] = int(tmp_id)

            ensemble_results.append(cur_result)

    mmcv.dump(ensemble_results, 'result.json')

if __name__ == '__main__':
    dt_result_1_path = 'json_files/mmdet-fcos-x101-42.5.json' # ensemble_file A
    dt_result_2_path = 'json_files/mmdet-r2-mask-43.6.json' # ensemble_file B
    gt_result_path = 'json_files/instances_val2017.json' # annotations

    dt_result_1 = mmcv.load(dt_result_1_path)
    dt_result_2 = mmcv.load(dt_result_2_path)
    gt_result = mmcv.load(gt_result_path)

    ########## confidence refinement ##########
    score_transform_1 = bandit_score_transform(dt_result_1, gt_result)
    score_transform_2 = bandit_score_transform(dt_result_2, gt_result)
    dt_result_1 = conf_refinement(dt_result_1, score_transform_1)
    dt_result_2 = conf_refinement(dt_result_2, score_transform_2)

    ensemble_results(dt_result_1, dt_result_2, gt_result)




