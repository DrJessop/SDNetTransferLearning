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


class MRI(Dataset):
    def __init__(self, device, dir, desired_height=160, desired_width=160, aug=False):
        super(MRI, self).__init__()
        self.dir = dir
        self.images = os.listdir(self.dir)
        self.desired_height = desired_height
        self.desired_width = desired_width
        self.device = device
        self.aug = aug

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
        if self.aug:
            sample = torch.Tensor(np.random.choice(shape[0], 9, replace=False)).long()
            im[sample] = im[sample].transpose(1, 2)
        return im.to(self.device)


def split_slices(image_batch):
    if len(image_batch.shape) == 4:
        image_batch = image_batch.unsqueeze(2)
    num_im1, num_im2, num_slices, height, width = image_batch.shape
    return image_batch.view(num_im1*num_im2*num_slices, 1, height, width)


def compute_loss(model, optimizer, batch, kl_loss, prior, reconstruction_loss, mean_absolute_error,
                 num_batches, train_mode=True, kl_loss_weight=0.001):

    batch = split_slices(batch)
    anatomy, reconstruction, modality_distribution = model(batch.float())

    kl_div = torch.mean(kl_divergence(modality_distribution, prior))

    kl_loss += kl_div.item()

    mae = mean_absolute_error(batch.float(), reconstruction)

    reconstruction_loss += mae.item()

    sample_z = modality_distribution.sample()
    dec_im = model.decoder(anatomy, sample_z)
    input_im = torch.cat((dec_im, anatomy), dim=1)
    modality_factor_reconstruct = model.encoder.mod(input_im)

    z_reconstruction_error = torch.mean(torch.abs(sample_z - modality_factor_reconstruct.mean))

    if train_mode:
        (kl_loss_weight*kl_div + mae + z_reconstruction_error).backward()
        optimizer.step()
        optimizer.zero_grad()
    num_batches += 1
    return kl_loss, reconstruction_loss, num_batches


def display_histogram(numpy_image):
    histogram = np.histogram(numpy_image)
    histogram = histogram[1][:-1], histogram[0]
    plt.bar(*histogram)
    plt.show()


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

        anatomy, reconstruction, _ = sdnet(test_im.float())

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


def viz_plot(viz, reconstruction_loss, kl_loss, reconstruction_loss_eval, kl_loss_eval, epoch):
    if epoch == 1:
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss]), win="loss", name="train_rec_loss")
        viz.line(X=np.array([epoch]), Y=np.array([kl_loss]), win="loss", name="train_kl_loss")
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss_eval]), win="loss", name="val_rec_loss")
        viz.line(X=np.array([epoch]), Y=np.array([kl_loss_eval]), win="loss", name="val_kl_loss")

    else:
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss]), win="loss", name="train_rec_loss",
                 update="append")
        viz.line(X=np.array([epoch]), Y=np.array([kl_loss]), win="loss", name="train_kl_loss",
                 update="append")
        viz.line(X=np.array([epoch]), Y=np.array([reconstruction_loss_eval]), win="loss", name="val_rec_loss",
                 update="append")
        viz.line(X=np.array([epoch]), Y=np.array([kl_loss_eval]), win="loss", name="val_kl_loss",
                 update="append")


def train_sdnet(data, model, epochs, kl_loss_weight=0.1, show_every=None):

    mean_absolute_error = nn.L1Loss(reduction="mean")

    t = trange(epochs, desc='Training progress...', leave=True)

    try:
        viz = visdom.Visdom(port=8097)
        visualize = True

    except:
        visualize = False

    optimizer = Adam(sdnet.parameters(), lr=1e-4)
    train_data, val_data = data
    num_plots_on_screen = 0

    modality_prior = Normal(loc=0.0, scale=1.0)
    for epoch in t:
        model.zero_grad()
        kl_loss = 0
        reconstruction_loss = 0
        num_batches = 0

        model.train()
        for batch in train_data:
            kl_loss, reconstruction_loss, num_batches = compute_loss(model, optimizer, batch, kl_loss, modality_prior,
                                                                     reconstruction_loss, mean_absolute_error,
                                                                     num_batches, train_mode=True,
                                                                     kl_loss_weight=kl_loss_weight)

        kl_loss = kl_loss/num_batches
        reconstruction_loss = reconstruction_loss/num_batches

        kl_loss_eval = 0
        reconstruction_loss_eval = 0
        num_batches = 0

        with torch.no_grad():
            model.eval()
            for batch in val_data:
                kl_loss_eval, reconstruction_loss_eval, num_batches = compute_loss(model, optimizer, batch, kl_loss,
                                                                                   modality_prior, reconstruction_loss,
                                                                                   mean_absolute_error, num_batches,
                                                                                   train_mode=False)

        if visualize:
            viz_plot(viz, reconstruction_loss, kl_loss, reconstruction_loss_eval, kl_loss_eval, epoch)

        if show_every is not None:
            if epoch % show_every == 0:
                test_slice(test_im_id=np.random.randint(0, len(val)), images=val, sdnet=sdnet)
                num_plots_on_screen += 1
            if num_plots_on_screen == 5:
                plt.clf()
                num_plots_on_screen = 0

        t.set_description("KL loss (train: {}, eval: {}), ".format(kl_loss, kl_loss_eval) +
                          "Rec loss (train: {}, eval: {}".format(reconstruction_loss, reconstruction_loss_eval))

    return model


def collate_function(batch):
    im = torch.empty(0, 1, 160, 160).to(gpu).float()
    for el in batch:
        el_reshaped = el.view(el.shape[0], 1, el.shape[1], el.shape[2]).float()
        im = torch.cat((im, el_reshaped))
    return im


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
    kgh = True
    scratch = True
    aug = False

    if kgh:
        image_dir = "/home/andrewg/PycharmProjects/assignments/merged_data/t2_full/KGH"
        if binary_threshold:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/KGH/sdnet_bin.pt"
        else:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/KGH/sdnet.pt"
        epochs = 100
    else:
        image_dir = "/home/andrewg/PycharmProjects/assignments/resampled/t2"
        if binary_threshold:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/ProstateX/sdnet_bin.pt"
        else:
            sdnet_file = "/home/andrewg/PycharmProjects/assignments/SDNetModels/ProstateX/sdnet.pt"
        epochs = 200

    p_images = MRI(device=gpu, dir=image_dir, aug=aug)
    n_train = int(0.8*len(p_images))
    train, val = random_split(p_images, lengths=(n_train, len(p_images) - n_train))
    train_loader = DataLoader(train, batch_size=batch_size, collate_fn=collate_function, shuffle=True)
    val_loader = DataLoader(val, batch_size=batch_size, collate_fn=collate_function, shuffle=True)

    sdnet = SDNet(n_a=4, n_z=8, device=gpu, binary_threshold=binary_threshold)

    if training:
        if not scratch:  # If I would like a pre-trained model loaded in
            sdnet.load_state_dict(torch.load(sdnet_file, map_location=gpu))

        train_sdnet((train_loader, val_loader), sdnet, epochs=epochs, kl_loss_weight=0.5, show_every=10)
        if save_model:
            torch.save(sdnet.state_dict(), sdnet_file)
    else:
        sdnet.load_state_dict(torch.load(sdnet_file, map_location=gpu))
        test_slice(test_im_id=np.random.randint(0, len(p_images)), images=p_images, sdnet=sdnet)

