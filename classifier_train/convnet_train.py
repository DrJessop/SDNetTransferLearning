import torch
from tqdm import trange
import torch.nn as nn
from torch.optim import Adam
import visdom
import numpy as np
import sys
sys.path.append("..")


def viz_plot(viz, nll_loss, nll_loss_eval, epoch):
    if epoch == 1:
        viz.line(X=np.array([epoch]), Y=np.array([nll_loss]), win="loss", name="train_nll_loss")
        viz.line(X=np.array([epoch]), Y=np.array([nll_loss_eval]), win="loss", name="val_nll_loss")

    else:
        viz.line(X=np.array([epoch]), Y=np.array([nll_loss]), win="loss", name="train_nll_loss", update="append")
        viz.line(X=np.array([epoch]), Y=np.array([nll_loss_eval]), win="loss", name="val_nll_loss", update="append")


def compute_loss(model, optimizer, images, target, loss_fn, train_mode):

    predicted = model(images)
    loss = loss_fn(predicted, target)
    hard_predicted = torch.argmax(torch.exp(predicted), dim=1)

    if train_mode:
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    correct = (hard_predicted == target).sum()
    incorrect = predicted.shape[0] - correct

    return loss.item(), correct.item(), incorrect.item()


def train(data_idx, images, labels, model, cnn_file, epochs, lr=0.0001, weight_decay=0.01, save_model=False):
    mean_nll_loss = nn.NLLLoss(reduction="mean")
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
        nll_loss = 0

        model.train()
        for batch_idx in train_data:
            batch_idx = batch_idx.reshape(batch_idx.shape[0]*batch_idx.shape[1]).long()
            image_batch, label_batch = images[batch_idx], labels[batch_idx]
            train_loss, _, _ = compute_loss(model, optimizer, image_batch, label_batch, mean_nll_loss, train_mode=True)
            nll_loss += train_loss

        nll_loss /= len(train_data)

        nll_loss_eval = 0
        eval_correct = 0
        eval_incorrect = 0
        with torch.no_grad():
            model.eval()
            for batch_idx in val_data:
                batch_idx = batch_idx.reshape(batch_idx.shape[0]*batch_idx.shape[1]).long()
                image_batch, label_batch = images[batch_idx], labels[batch_idx]
                eval_loss, correct, incorrect = compute_loss(model, optimizer, image_batch, label_batch,
                                                             mean_nll_loss, train_mode=False)
                nll_loss_eval += eval_loss
                eval_correct += correct
                eval_incorrect += incorrect

        nll_loss_eval /= len(val_data)
        eval_accuracy = eval_correct/(eval_correct + eval_incorrect)
        if visualize:
            viz_plot(viz, nll_loss, nll_loss_eval, epoch)

        if save_model and nll_loss_eval < lowest_loss:
            torch.save(model.state_dict(), cnn_file)
            lowest_loss = nll_loss_eval

        t.set_description("NLL loss on all batches (cnn_train: {:3f}, eval: {:3f}, acc: {:3f}), ".format(nll_loss,
                          nll_loss_eval, eval_accuracy))
