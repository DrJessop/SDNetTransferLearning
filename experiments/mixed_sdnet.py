import torch
import sys
sys.path.append("..")
from utils.config import get_config
from utils.confusion_matrix import cm
from utils.normalize import normalize
from utils.bootstrap import bootstrap_auc
from models.CAM_CNN import CAM_CNN
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix as conf_mat


def get_metric(probs, labels, verbose=False):
    binary_preds = (np.array(probs) > 0.5) * 1
    TN, FP, FN, TP = conf_mat(labels, binary_preds).ravel()
    fpr, tpr, thresholds = metrics.roc_curve(labels, probs, pos_label=1)
    acc = metrics.accuracy_score(labels, binary_preds)
    auc = metrics.auc(fpr, tpr)
    sens = TP / (TP + FN)
    spes = TN / (TN + FP)
    if verbose:
        print("Accuracy:{}, sensitivity :{}, specificity: {}, AUC:{}".format(acc, sens,spes,auc))
    return acc, sens, spes, auc


if __name__ == "__main__":
    conf = get_config()
    cnn = CAM_CNN(conf.shape, conf.num_channels).to(conf.device)
    cnn.load_state_dict(torch.load(conf.model_src))

    p_images_hold_out = torch.load(conf.test_images).to(conf.device)
    p_labels_hold_out = torch.load(conf.test_labels).to(conf.device)

    # p_images_hold_out = p_images_hold_out[p_images_hold_out.shape[0]//2:]
    # p_labels_hold_out = p_labels_hold_out[p_labels_hold_out.shape[0]//2:]

    for idx in range(p_images_hold_out.shape[0]):
        normalize(p_images_hold_out[idx])

    counter = 0

    confusion_matrix = np.zeros((2, 2))
    inc = p_images_hold_out.shape[0]//5
    predicted_total = torch.empty(0).float()
    with torch.no_grad():
        cnn.eval()
        for idx in range(0, p_images_hold_out.shape[0], inc):
            end = min(p_images_hold_out.shape[0], idx + inc)
            predicted = cnn(p_images_hold_out[idx:end])
            predicted_total = torch.cat((predicted_total, predicted[:, 1]))
            target = p_labels_hold_out[idx:end]
            cm_current_batch = cm(target, predicted)
            confusion_matrix = confusion_matrix + cm_current_batch
            counter += 1

        get_metric(predicted_total, p_labels_hold_out, verbose=True)

    accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1])/confusion_matrix.ravel().sum()
    print("The accuracy is {}".format(accuracy))

    fig = plt.figure()
    ax = fig.add_subplot()

    bootstrap_auc(p_labels_hold_out.cpu().numpy(), predicted_total.cpu().numpy(), ax)
    plt.savefig("/home/andrewg/PycharmProjects/merged_data/images/magic.svg")
    # plt.show()

