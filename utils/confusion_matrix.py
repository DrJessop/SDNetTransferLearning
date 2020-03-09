from sklearn.metrics import confusion_matrix
import torch


def cm(target_tensor, predicted_tensor):
    predicted_tensor = torch.argmax(predicted_tensor, dim=1).numpy()
    target_tensor = target_tensor.numpy()

    return confusion_matrix(target_tensor, predicted_tensor)