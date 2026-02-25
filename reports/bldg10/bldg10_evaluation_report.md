# Building 10 Evaluation Report

## Dataset Summary

- Samples: 2089
- AP features: 178
- IMU features: 10
- Total features: 188
- Room classes: 47
- Floor classes: 2

## Room Classification (Cross-Validation)

| Model | Accuracy (mean+/-std) | Macro F1 (mean+/-std) |
|---|---|---|
| KNN | 0.9382 +/- 0.0152 | 0.9364 +/- 0.0151 |
| MLP | 0.9646 +/- 0.0100 | 0.9642 +/- 0.0100 |

## Holdout Results

### Room Classification

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|---|
| KNN | 0.9402 | 0.9457 | 0.9396 | 0.9389 |
| MLP | 0.9545 | 0.9603 | 0.9541 | 0.9544 |

### Floor Classification (KNN)

| Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---|---|
| 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Visualizations

Room class distribution:

![](plots/room_class_distribution.png)

Room confusion matrix (normalized):

![](plots/room_confusion_matrix_mlp.png)

Floor confusion matrix (normalized):

![](plots/floor_confusion_matrix_knn.png)
