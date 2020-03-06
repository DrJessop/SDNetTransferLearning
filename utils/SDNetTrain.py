import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import trange
import visdom
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence


def normalize(im):

    mins = [im[idx].min() for idx in range(len(im))]
    maxes = [im[idx].max() for idx in range(len(im))]

    for idx in range(len(im)):
        min_val = mins[idx]
        max_val = maxes[idx]

        if min_val == max_val:
            im[idx] = torch.zeros(im[idx].shape)
        else:
            im[idx] = 2*(im[idx] - min_val)/(max_val - min_val) - 1


def compute_loss(model, optimizer, batch, kl_loss, prior, reconstruction_loss, z_rec_loss, mean_absolute_error,
                 train_mode=True, kl_loss_weight=0.001):

    anatomy, reconstruction, modality_distribution, z = model(batch.float())

    kl_div = torch.mean(kl_divergence(modality_distribution, prior))
    kl_loss += kl_div.item()

    mae = mean_absolute_error(batch.float(), reconstruction)
    reconstruction_loss += mae.item()

    _, _, _, z2 = model(reconstruction)
    z_rec_mae = mean_absolute_error(z, z2)
    z_rec_loss += z_rec_mae.item()

    if train_mode:
        (kl_loss_weight*kl_div + mae + z_rec_mae).backward()
        optimizer.step()
        optimizer.zero_grad()

    return kl_loss, reconstruction_loss, z_rec_loss


def test_slice(test_im_id, images, sdnet):

    with torch.no_grad():
        # Plotting the reconstruction of a specific slice
        sdnet.eval()
        test_im = images[test_im_id]
        if len(test_im.shape) == 3:
            test_im = test_im[0].unsqueeze(0).unsqueeze(0)
        else:
            test_im = images[test_im_id][0][1]
            test_im = test_im.view(1, 1, *test_im.shape)

        test_im = test_im.float()

        anatomy, reconstruction, _, _ = sdnet(test_im.float())

        plt.figure()
        num_cols = anatomy.shape[1] // 2
        _, axarr = plt.subplots(2, num_cols)

        for i in range(2):
            for j in range(num_cols):
                axarr[i, j].imshow(anatomy[0][j + num_cols*i].detach().cpu().numpy(), cmap="gray")
                axarr[i, j].set_axis_off()
        plt.show()

        reconstruction = reconstruction[0][0].detach().cpu().numpy()

        test_im = test_im[0][0].cpu().numpy()

        plt.figure()
        _, axarr = plt.subplots(1, 2)

        axarr[0].imshow(test_im, cmap="gray")
        axarr[1].imshow(reconstruction, cmap="gray")
        axarr[0].set_axis_off()
        axarr[1].set_axis_off()
        plt.show()

    return


def viz_plot(viz, reconstruction_loss, kl_loss, z_rec_loss, reconstruction_loss_eval, kl_loss_eval, z_rec_loss_eval,
             epoch):
    if epoch == 1:
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss]), win="loss", name="train_rec_loss")
        viz.line(X=np.array([epoch]), Y=np.array([kl_loss]), win="loss", name="train_kl_loss")
        viz.line(X=np.array([epoch]), Y=np.array([z_rec_loss]), win="loss", name="train_zrec_loss")
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss_eval]), win="loss", name="val_rec_loss")
        viz.line(X=np.array([epoch]), Y=np.array([kl_loss_eval]), win="loss", name="val_kl_loss")
        viz.line(X=np.array([epoch]), Y=np.array([z_rec_loss_eval]), win="loss", name="val_zrec_loss")

    else:
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss]), win="loss", name="train_rec_loss",
                 update="append")
        viz.line(X=np.array([epoch]), Y=np.array([kl_loss]), win="loss", name="train_kl_loss",
                 update="append")
        viz.line(X=np.array([epoch]), Y=np.array([z_rec_loss]), win="loss", name="train_zrec_loss",
                 update="append")
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss_eval]), win="loss", name="val_rec_loss",
                 update="append")
        viz.line(X=np.array([epoch]), Y=np.array([kl_loss_eval]), win="loss", name="val_kl_loss",
                 update="append")
        viz.line(X=np.array([epoch]), Y=np.array([z_rec_loss_eval]), win="loss", name="val_zrec_loss",
                 update="append")


def train(data, model, epochs, sdnet_file, kl_loss_weight=0.1, show_every=None, save_model=False, lr=0.0001):

    mean_absolute_error = nn.L1Loss(reduction="mean")

    t = trange(epochs, desc='Training progress...', leave=True)

    best_rec_loss = np.inf

    try:
        viz = visdom.Visdom(port=8097)
        visualize = True

    except:
        visualize = False

    optimizer = Adam(model.parameters(), lr=lr)
    train_data, val_data = data

    modality_prior = Normal(loc=0.0, scale=1.0)
    for epoch in t:
        model.zero_grad()
        kl_loss = 0
        reconstruction_loss = 0
        z_rec_loss = 0

        model.train()
        for batch in train_data:
            kl_loss, reconstruction_loss, z_rec_loss = compute_loss(model, optimizer, batch, kl_loss,
                                                                    modality_prior, reconstruction_loss,
                                                                    z_rec_loss, mean_absolute_error,
                                                                    train_mode=True,
                                                                    kl_loss_weight=kl_loss_weight)

        kl_loss = kl_loss*kl_loss_weight/len(train_data)
        reconstruction_loss = reconstruction_loss/len(train_data)
        z_rec_loss = z_rec_loss/len(train_data)

        kl_loss_eval = 0
        reconstruction_loss_eval = 0
        z_rec_loss_eval = 0

        with torch.no_grad():
            model.eval()
            for batch in val_data:
                kl_loss_eval, reconstruction_loss_eval, z_rec_loss_eval = compute_loss(model, optimizer,
                                                                                       batch, kl_loss_eval,
                                                                                       modality_prior,
                                                                                       reconstruction_loss_eval,
                                                                                       z_rec_loss_eval,
                                                                                       mean_absolute_error,
                                                                                       train_mode=False)
        kl_loss_eval = kl_loss_eval*kl_loss_weight/len(val_data)
        reconstruction_loss_eval = reconstruction_loss_eval/len(val_data)
        z_rec_loss_eval = z_rec_loss_eval/len(val_data)

        if visualize:
            viz_plot(viz, reconstruction_loss, kl_loss, z_rec_loss, reconstruction_loss_eval, kl_loss_eval,
                     z_rec_loss_eval, epoch)

        total_loss = kl_loss_eval + reconstruction_loss_eval + z_rec_loss_eval
        if show_every is not None:
            if save_model and total_loss < best_rec_loss:
                torch.save(model.state_dict(), sdnet_file)
                best_rec_loss = reconstruction_loss_eval
            if epoch % show_every == 0:
                test_slice(test_im_id=np.random.randint(0, len(val_data)), images=val_data, sdnet=model)

        t.set_description("KL loss (sdnet_train: {}, eval: {}), ".format(kl_loss, kl_loss_eval) +
                          "Rec loss (sdnet_train: {}, eval: {}), ".format(reconstruction_loss, reconstruction_loss_eval) +
                          "Zrec loss (sdnet_train: {}, eval: {})".format(z_rec_loss, z_rec_loss_eval))

    return model

