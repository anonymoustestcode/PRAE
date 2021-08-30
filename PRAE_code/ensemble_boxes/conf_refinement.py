import mmcv
import math

def conf_refinement(dt_result, value):
    '''
    :param dt_result: predictions
    :param value: score transform list
    :return: refined predictions
    '''

    ##### conf refinement #####
    aver = value[0]
    top = value[1]

    refined_results = []
    for tmp in dt_result:
        if tmp['score'] < 0.04:
            continue
        index = min(tmp['score'] // 0.04, len(top))
        # print(index, top[int(index)-1])
        # input()
        # print(index, len(top), len(aver))
        tmp['score'] = tmp['score'] * top[int(index) - 1] / (
        aver[int(index) - 1])  # * (math.exp((distrib[int(index)-1]-m_value)**2)))
        # tmp['score'] = top[int(index)-1]
        refined_results.append(tmp)

    return refined_results