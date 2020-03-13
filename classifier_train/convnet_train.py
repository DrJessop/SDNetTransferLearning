import torch
from tqdm import trange
import torch.nn as nn
from torch.optim import Adam
import visdom
import numpy as np
from sklearn import metrics
import math
import sys
sys.path.append("..")


def viz_plot(viz, nll_loss, nll_loss_eval, acc, epoch):
    if epoch == 1:
        viz.line(X=np.array([epoch]), Y=np.array([nll_loss]), win="loss", name="train_loss")
    else:
        viz.line(X=np.array([epoch]), Y=np.array([nll_loss]), win="loss", name="train_loss", update="append")

    viz.line(X=np.array([epoch]), Y=np.array([nll_loss_eval]), win="loss", name="val_loss", update="append")
    viz.line(X=np.array([epoch]), Y=np.array([acc]), win="loss", name="val_auc", update="append")


def compute_loss(model, optimizer, images, target, loss_fn, train_mode):

    predicted = model(images)
    loss = loss_fn(predicted, target)

    auc = None

    if train_mode:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    else:
        fpr, tpr, thresholds = metrics.roc_curve(target.cpu().numpy(), predicted[:, 1].cpu().numpy(), pos_label=1)
        auc = metrics.auc(fpr, tpr)
        if math.isnan(auc):
            auc = 0

    return loss.item(), auc


def train(data_idx, images, labels, model, cnn_file, epochs, lr=0.0001, weight_decay=0.01, save_model=False):
    loss_fn = nn.CrossEntropyLoss(reduction="mean")
    t = trange(epochs, desc='Training progress...', leave=True)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_data, val_data = data_idx

    try:
        viz = visdom.Visdom(port=8097)
        visualize = True
    except:
        visualize = False

    lowest_loss = np.inf
    for epoch in t:
        model.zero_grad()
        loss = 0

        model.train()
        for batch_idx in train_data:
            batch_idx = batch_idx.view(-1).long()
            image_batch, label_batch = images[batch_idx], labels[batch_idx]

            train_loss, _ = compute_loss(model, optimizer, image_batch, label_batch, loss_fn, train_mode=True)
            loss += train_loss

        loss /= len(train_data)

        loss_eval = 0
        avg_auc = 0
        with torch.no_grad():
            model.eval()
            for batch_idx in val_data:
                batch_idx = batch_idx.view(-1).long()
                image_batch, label_batch = images[batch_idx], labels[batch_idx]

                eval_loss, auc = compute_loss(model, optimizer, image_batch, label_batch, loss_fn, train_mode=False)
                loss_eval += eval_loss
                avg_auc += auc

        loss_eval /= len(val_data)
        avg_auc /= len(val_data)

        if visualize:
            viz_plot(viz, loss, loss_eval, avg_auc, epoch)

        if save_model and loss_eval < lowest_loss:
            torch.save(model.state_dict(), cnn_file)
            lowest_loss = loss_eval

        t.set_description("Loss on all batches (cnn_train: {:3f}, eval: {:3f}, auc: {:3f}), ".format(loss,
                          loss_eval, avg_auc))
