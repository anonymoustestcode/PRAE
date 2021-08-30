# Probabilistic Ranking-Aware Ensembles (PRAE).

![image](https://github.com/UserUnknown0123/neurips2021)

## Results  

**FCOS (ResNext101-FPN) with 42.5% mAP**  
**&**  
**Mask R-CNN (Res2Net101-FPN with 43.6% mAP**  

| Methods | IOU threshold | sigma | weight | mAPl<br> (COCO val) |
| :-: | :-: | :-: | :-: | :-: |
| NMS | 0.65 | None | [2, 3] | 44.6 |
| Soft-NMS | 0.7 | 0.1 | [4, 5] | 44.7 |
| NMW | 0.7 | None | [3, 4] | 45.7 |
| WBF | 0.7 | None | [2, 3] | 45.2 |
| **PRAE** | **0.7** | **None** | **[1, 1]** | **46.7** |

## Prerequisites 

* Python 3.6+

* mmcv >= 0.4.0

* numpy >= 1.16

* cocoapi


## Usage
```bash
# implement PRAE
python ensemble-2-models.py   

# calculate mAP on COCO2017 validation set
python evaluate.py 
```
