import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from tqdm import trange
import visdom
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt


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
    # import SimpleITK as sitk
    # from utils.normalize import normalize
    with torch.no_grad():

        # t1_5 = sitk.ReadImage("/home/andrewg/PycharmProjects/merged_data/t2_full/ProstateX/ProstateX-0000.nrrd")
        # t1_5 = torch.from_numpy(sitk.GetArrayFromImage(t1_5).astype(np.int64))
        # plt.imshow(t1_5[t1_5.shape[0]//2], cmap="gray")
        # plt.axis("off")
        # plt.savefig("/home/andrewg/PycharmProjects/images/1_5t.eps")
        # for idx in range(len(t1_5)):
        #     normalize(t1_5[idx])

        # _, _, _, z = sdnet(t1_5.unsqueeze(1).cuda(1).float())
        # z = z[0].unsqueeze(0)
        # print(z.shape)

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
        # print(anatomy.shape)

        # reconstruction = sdnet.decoder(anatomy, z)

        plt.figure()
        num_rows = 2
        num_cols = 2
        _, axarr = plt.subplots(num_rows, num_cols)

        for i in range(num_rows):
            for j in range(num_cols):
                if j + num_cols*i == anatomy.shape[1]:
                    break
                axarr[i, j].imshow(anatomy[0][j + num_cols*i].detach().cpu().numpy(), cmap="gray")
                axarr[i, j].set_axis_off()

        # plt.savefig("/home/andrewg/PycharmProjects/images/anat.eps")
        plt.axis("off")
        plt.show()
        plt.clf()

        reconstruction = reconstruction[0][0].detach().cpu().numpy()

        test_im = test_im[0][0].cpu().numpy()

        plt.figure()
        _, axarr = plt.subplots(1, 2)

        axarr[0].imshow(test_im, cmap="gray")
        axarr[1].imshow(reconstruction, cmap="gray")
        axarr[0].set_axis_off()
        axarr[1].set_axis_off()
        # plt.savefig("/home/andrewg/PycharmProjects/images/rec.eps")
        plt.show()
        # plt.clf()

    return


def viz_plot(viz, reconstruction_loss, kl_loss, z_rec_loss, reconstruction_loss_eval, kl_loss_eval, z_rec_loss_eval,
             epoch):
    if epoch == 1:
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss]), win="loss", name="train_rec_loss")
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


def train(data, model, epochs, sdnet_file, val_images, kl_loss_weight=0.1, show_every=None, save_model=False,
          lr=0.0001):

    mean_absolute_error = nn.L1Loss(reduction="mean")

    t = trange(epochs, desc='Training progress...', leave=True)

    lowest_loss = np.inf

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

        kl_loss /= len(train_data)
        reconstruction_loss /= len(train_data)
        z_rec_loss /= len(train_data)

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

        kl_loss_eval /= len(val_data)
        reconstruction_loss_eval /= len(val_data)
        z_rec_loss_eval /= len(val_data)

        if visualize:
            viz_plot(viz, reconstruction_loss, kl_loss, z_rec_loss, reconstruction_loss_eval, kl_loss_eval,
                     z_rec_loss_eval, epoch)

        total_loss = kl_loss_weight*kl_loss_eval + reconstruction_loss_eval + z_rec_loss_eval

        if save_model and total_loss < lowest_loss:
            torch.save(model.state_dict(), sdnet_file)
            lowest_loss = total_loss

        if show_every is not None:
            if epoch % show_every == 0:
                test_slice(test_im_id=np.random.randint(0, len(val_data)), images=val_images, sdnet=model)

        t.set_description("KL loss (sdnet_train: {}, eval: {}), ".format(kl_loss, kl_loss_eval) +
                          "Rec loss (sdnet_train: {}, eval: {}), ".format(reconstruction_loss, reconstruction_loss_eval) +
                          "Zrec loss (sdnet_train: {}, eval: {})".format(z_rec_loss, z_rec_loss_eval))

    return model



