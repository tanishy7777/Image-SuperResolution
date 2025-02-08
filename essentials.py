`import torch
from einops import rearrange
import torch.nn as nn
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearModel(nn.Module):
    def __init__(self, in_features, out_features):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.linear(x)


def create_coordinate_map(img, factor=1):
    num_channels, height, width = img.shape
    height *= factor
    width *= factor

    w_coords = torch.arange(width).repeat(height, 1)
    h_coords = torch.arange(height).repeat(width, 1).t()
    w_coords = w_coords.reshape(-1)
    h_coords = h_coords.reshape(-1)

    X = torch.stack([h_coords, w_coords], dim=1).float().to(device)
    Y = rearrange(img, 'c h w -> (h w) c').float()
    return X, Y


def create_rff_features(X, num_features, sigma):
    from sklearn.kernel_approximation import RBFSampler
    rff = RBFSampler(n_components=num_features, gamma=1 / (2 * sigma ** 2))
    X = X.cpu().numpy()
    X = rff.fit_transform(X)
    return torch.tensor(X, dtype=torch.float32).to(device)


def train(net, lr, X, Y, epochs, verbose=True):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = net(X)

        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch} loss: {loss.item():.6f}")

    return loss.item()


def plot_reconstructed_and_original_image(original_img, reconstructed_img, title):
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.imshow(reconstructed_img)
    ax0.set_title("Reconstructed Image")

    ax1.imshow(original_img)
    ax1.set_title("Original Image")

    for a in [ax0, ax1]:
        a.axis("off")

    fig.suptitle(title, y=0.9)
    plt.tight_layout()


def plot_super_resolution_and_low_resolution_image(low_resolution_image, super_resolution_img, title):
    fig = plt.figure(figsize=(6, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])

    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    ax0.imshow(super_resolution_img)
    ax0.set_title("Super Resolution Image: 400x400")

    ax1.imshow(low_resolution_image)
    ax1.set_title("Low Resolution Image: 200x200")

    for a in [ax0, ax1]:
        a.axis("off")

    fig.suptitle(title, y=0.9)
    plt.tight_layout()


def calculate_rmse(original, trained):
    return np.sqrt(np.mean((original - trained) ** 2))


def calculate_psnr(original_image, reconstructed_image):
    return 20 * np.log10(1.0 / calculate_rmse(original_image, reconstructed_image))


def calculate_snr(original_audio, reconstructed_audio):
    return 10 * np.log10(np.sum(original_audio ** 2) / np.sum((original_audio - reconstructed_audio) ** 2))


def audio_plot(audio, sr, clr, tl):
    plt.figure(figsize=(15, 4))
    plt.plot(audio, color=clr, alpha=0.7)
    plt.xticks(np.arange(0, audio.shape[0], sr), np.arange(0, audio.shape[0] / sr, 1))
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(tl)
    plt.grid()
    plt.show()
`