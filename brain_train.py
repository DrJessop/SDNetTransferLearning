import SimpleITK as sitk
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
import os
from SDNet import SDNet
from torch.optim import Adam
from tqdm import trange
import visdom
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


class Brain(Dataset):
    def __init__(self, device, dir, desired_slices, desired_height, desired_width):
        super(Brain, self).__init__()
        self.dir = dir
        self.images = os.listdir(self.dir)
        self.desired_slices = desired_slices//2
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.device = device

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im = sitk.ReadImage("{}/{}".format(self.dir, self.images[idx]))
        im = sitk.GetArrayFromImage(im).astype(np.float64)
        im = torch.from_numpy(im)
        shape = im.shape
        im = im[:, (shape[1] - self.desired_height) // 2:(shape[1] - self.desired_height) // 2 + self.desired_height]
        im = im[:, :,  (shape[2] - self.desired_width) // 2:(shape[2] - self.desired_width) // 2 + self.desired_width]
        normalize(im)
        midpoint = im.shape[0] // 2
        im = im[midpoint - self.desired_slices: midpoint + self.desired_slices]
        return im.to(self.device)


def split_slices(image_batch):
    if len(image_batch.shape) == 4:
        image_batch = image_batch.unsqueeze(2)
    num_im1, num_im2, num_slices, height, width = image_batch.shape
    return image_batch.view(num_im1*num_im2*num_slices, 1, height, width)


def compute_loss(model, optimizer, batch, kl_loss, prior, reconstruction_loss, z_rec_loss, mean_absolute_error,
                 num_batches, train_mode=True, kl_loss_weight=0.001):

    anatomy, reconstruction, modality_distribution, z = model(batch.float())

    kl_div = torch.mean(kl_divergence(modality_distribution, prior))
    kl_loss += kl_div.item()

    mae = mean_absolute_error(batch.float(), reconstruction)
    reconstruction_loss += mae.item()

    _, _, _, z2 = model(reconstruction)
    z_rec_mae = mean_absolute_error(z, z2)
    z_rec_loss += z_rec_mae.item()

    if train_mode:
        (kl_loss_weight*kl_div + mae + z_rec_loss).backward()
        optimizer.step()
        optimizer.zero_grad()

    return kl_loss, reconstruction_loss, z_rec_loss, num_batches + 1


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


def train_sdnet(data, model, epochs, kl_loss_weight=0.1, show_every=None, save_model=False):

    mean_absolute_error = nn.L1Loss(reduction="mean")

    t = trange(epochs, desc='Training progress...', leave=True)

    best_rec_loss = np.inf

    try:
        viz = visdom.Visdom(port=8097)
        visualize = True

    except:
        visualize = False

    optimizer = Adam(sdnet.parameters(), lr=1e-4)
    train_data, val_data = data

    modality_prior = Normal(loc=0.0, scale=1.0)
    for epoch in t:
        model.zero_grad()
        kl_loss = 0
        reconstruction_loss = 0
        z_rec_loss = 0
        num_batches = 0

        model.train()
        for batch in train_data:
            kl_loss, reconstruction_loss, z_rec_loss, num_batches = compute_loss(model, optimizer, batch, kl_loss,
                                                                                 modality_prior, reconstruction_loss,
                                                                                 z_rec_loss, mean_absolute_error,
                                                                                 num_batches, train_mode=True,
                                                                                 kl_loss_weight=kl_loss_weight)

        kl_loss = kl_loss/num_batches
        reconstruction_loss = reconstruction_loss/num_batches
        z_rec_loss = z_rec_loss/num_batches

        kl_loss_eval = 0
        reconstruction_loss_eval = 0
        z_rec_loss_eval = 0
        num_batches = 0

        with torch.no_grad():
            model.eval()
            for batch in val_data:
                kl_loss_eval, reconstruction_loss_eval, z_rec_loss_eval, num_batches = compute_loss(model, optimizer,
                                                                                                    batch, kl_loss_eval,
                                                                                                    modality_prior,
                                                                                                    reconstruction_loss_eval,
                                                                                                    z_rec_loss_eval,
                                                                                                    mean_absolute_error,
                                                                                                    num_batches,
                                                                                                    train_mode=False)
        kl_loss_eval = kl_loss_eval/num_batches
        reconstruction_loss_eval = reconstruction_loss_eval/num_batches
        z_rec_loss_eval = z_rec_loss_eval/num_batches

        if visualize:
            viz_plot(viz, reconstruction_loss, kl_loss, z_rec_loss, reconstruction_loss_eval, kl_loss_eval,
                     z_rec_loss_eval, epoch)

        if show_every is not None:
            # if save_model and reconstruction_loss_eval < best_rec_loss:
                # torch.save(sdnet.state_dict(), sdnet_file)
                #  best_rec_loss = reconstruction_loss_eval
            if epoch % show_every == 0:
                test_slice(test_im_id=np.random.randint(0, len(val)), images=val, sdnet=sdnet)

        t.set_description("KL loss (train: {}, eval: {}), ".format(kl_loss, kl_loss_eval) +
                          "Rec loss (train: {}, eval: {}), ".format(reconstruction_loss, reconstruction_loss_eval) +
                          "Zrec loss (train: {}, eval: {})".format(z_rec_loss, z_rec_loss_eval))

    return model


def collate_function(batch):
    im = torch.empty(0, 1, HEIGHT, WIDTH).to(gpu).float()
    for el in batch:
        el_reshaped = el.view(el.shape[0], 1, el.shape[1], el.shape[2]).float()
        im = torch.cat((im, el_reshaped))
    return im


def convert_image(sdnet, convert="tot2"):

    with torch.no_grad():
        sdnet.eval()
        t1 = "/home/andrewg/PycharmProjects/assignments/merged_data/brain/combined/case012_T1.nii.gz"
        t2 = "/home/andrewg/PycharmProjects/assignments/merged_data/brain/combined/case012_T2.nii.gz"

        t1 = sitk.ReadImage(t1)
        t1 = sitk.GetArrayFromImage(t1).astype(np.float64)
        t1 = torch.from_numpy(t1)
        shape = t1.shape
        t1 = t1[:, (shape[1] - HEIGHT) // 2:(shape[1] - HEIGHT) // 2 + HEIGHT]
        t1 = t1[:, :, (shape[2] - WIDTH) // 2:(shape[2] - WIDTH) // 2 + WIDTH]
        normalize(t1)
        midpoint = t1.shape[0] // 2
        t1 = t1[midpoint - SLICES//2: midpoint + SLICES//2]
        t1 = t1[t1.shape[0] // 2].unsqueeze(0).unsqueeze(0).to(gpu).float()

        t2 = sitk.ReadImage(t2)
        t2 = sitk.GetArrayFromImage(t2).astype(np.float64)
        t2 = torch.from_numpy(t2)
        shape = t2.shape
        t2 = t2[:, (shape[1] - HEIGHT) // 2:(shape[1] - HEIGHT) // 2 + HEIGHT]
        t2 = t2[:, :, (shape[2] - WIDTH) // 2:(shape[2] - WIDTH) // 2 + WIDTH]
        normalize(t2)
        midpoint = t2.shape[0] // 2
        t2 = t2[midpoint - SLICES // 2: midpoint + SLICES // 2]
        t2 = t2[t2.shape[0] // 2].unsqueeze(0).unsqueeze(0).to(gpu).float()

        if convert == "tot2":
            anatomy, _, _ = sdnet(t1)
            _, _, modality_distribution = sdnet(t2)
            # sample = modality_distribution.sample()
            sample = torch.rand_like(torch.zeros(1, 8)).to(gpu).float()
            reconstruction = sdnet.decoder(anatomy, sample)

            plt.figure()
            num_cols = anatomy.shape[1] // 2
            _, axarr = plt.subplots(2, num_cols)
            test_im = t1[0][0].cpu().numpy()

        elif convert == "tot1":
            anatomy, _, _ = sdnet(t2)
            _, _, modality_distribution = sdnet(t1)
            sample = modality_distribution.sample()
            reconstruction = sdnet.decoder(anatomy, sample)

            plt.figure()
            num_cols = anatomy.shape[1] // 2
            _, axarr = plt.subplots(2, num_cols)
            test_im = t2[0][0].cpu().numpy()
        else:
            raise ValueError("Acceptable conversions are tot2 or tot1")

        for i in range(2):
            for j in range(num_cols):
                axarr[i, j].imshow(anatomy[0][j + num_cols*i].detach().cpu().numpy(), cmap="gray")
                axarr[i, j].set_axis_off()
        plt.show()

        reconstruction = reconstruction[0][0].detach().cpu().numpy()

        plt.figure()
        _, axarr = plt.subplots(1, 2)

        axarr[0].imshow(test_im, cmap="gray")
        axarr[1].imshow(reconstruction, cmap="gray")
        axarr[0].set_axis_off()
        axarr[1].set_axis_off()
        plt.show()

    return


if __name__ == "__main__":

    seed = 0
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    gpu = torch.device("cuda:1")
    batch_size = 2
    binary_threshold = True
    training = True
    save_model = True
    scratch = True
    SLICES, HEIGHT, WIDTH = (20, 192, 192)
    train_on = "combined"

    epochs = 200
    if train_on == "T1":
        image_dir = "/home/andrewg/PycharmProjects/assignments/merged_data/brain/T1_brain"
        if binary_threshold:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/Brain/T1/sdnet_bin.pt"
        else:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/Brain/T1/sdnet.pt"

    elif train_on == "T2":
        image_dir = "/home/andrewg/PycharmProjects/assignments/merged_data/brain/T2_brain"
        if binary_threshold:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/Brain/T2/sdnet_bin.pt"
        else:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/Brain/T2/sdnet.pt"
    elif train_on == "combined":
        image_dir = "/home/andrewg/PycharmProjects/assignments/merged_data/brain/combined"
        if binary_threshold:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/Brain/combined/sdnet_bin_T1T2.pt"
        else:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/Brain/combined/sdnet.pt"
    else:
        raise FileNotFoundError

    p_images = Brain(device=gpu, dir=image_dir, desired_slices=SLICES, desired_height=HEIGHT,
                     desired_width=WIDTH)
    n_train = int(0.8*len(p_images))
    train, val = random_split(p_images, lengths=(n_train, len(p_images) - n_train))
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_function, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_function, shuffle=True)

    sdnet = SDNet(n_a=4, n_z=8, device=gpu, shape=(HEIGHT, WIDTH), binary_threshold=binary_threshold)

    if training:
        if not scratch:  # If I would like a pre-trained model loaded in
            sdnet.load_state_dict(torch.load(sdnet_file, map_location=gpu))

        train_sdnet((train_loader, val_loader), sdnet, epochs=epochs, kl_loss_weight=0.01, show_every=10,
                    save_model=save_model)
    else:
        sdnet.load_state_dict(torch.load(sdnet_file, map_location=gpu))
        # test_slice(test_im_id=np.random.randint(0, len(p_images)), images=p_images, sdnet=sdnet)
        convert_image(sdnet, convert="tot2")
