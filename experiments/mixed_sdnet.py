import torch
import sys
sys.path.append("..")
from utils.config import get_config
from utils.normalize import normalize
from models.CAM_CNN import CAM_CNN
from sklearn import metrics
import math


if __name__ == "__main__":
    conf = get_config()
    cnn = CAM_CNN(conf.shape, conf.num_channels).to(conf.device)
    cnn.load_state_dict(conf.model_src)

    p_images_hold_out = torch.load(conf.test_images)
    p_labels_hold_out = torch.load(conf.test_labels)

    for idx in range(p_images_hold_out.shape[0]):
        normalize(p_images_hold_out[idx])

    avg_auc = 0
    counter = 0

    with torch.no_grad():
        cnn.eval()
        for idx in range(0, p_images_hold_out.shape[0], 100):
            end = min(p_images_hold_out.shape[0], idx + 100)
            predicted = cnn(p_images_hold_out[idx:end])[:, 1]
            target = p_labels_hold_out[idx:end]
            fpr, tpr, _ = metrics.roc_curve(target.cpu().numpy(), predicted.cpu().numpy())
            auc = metrics.auc(fpr, tpr)
            if math.isnan(auc):
                auc = 0
            avg_auc += auc
            counter += 1

    avg_auc /= counter
    print("The AUC is {}".format(avg_auc))
