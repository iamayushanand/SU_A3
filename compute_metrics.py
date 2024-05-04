from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np

SCORE_FILE = "custom_dataset_scores.txt"
labels = []
scores = []

with open(SCORE_FILE, "r") as f:
    for line in f:
        file_name = ' '.join(line.split()[:-1])
        score = float(line.split()[-1])
        scores.append(score)
        label = 1 if "Real/" in line else 0
        labels.append(label)
        
def get_eer(scores, labels):
    fpr, tpr, threshold = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.absolute((fnr - fpr)))
    eer_threshold = threshold[eer_index]
    EER = (fpr[eer_index]+fnr[eer_index])/2
    return EER

print("The EER is:", get_eer(scores, labels))
#0.44
print("The ROC-AUC is:", roc_auc_score(labels, scores))
#0.609

