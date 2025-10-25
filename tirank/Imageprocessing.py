import matplotlib.pyplot as plt
from PIL import Image

import torch
import timm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from itertools import repeat
import collections.abc
import os

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

class ConvStem(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=768, norm_layer=None, flatten=True):
        super().__init__()

        assert patch_size == 4
        assert embed_dim % 8 == 0

        to_2tuple = _ntuple(2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten


        stem = []
        input_dim, output_dim = 3, embed_dim // 8
        for l in range(2):
            stem.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=2, padding=1, bias=False))
            stem.append(nn.BatchNorm2d(output_dim))
            stem.append(nn.ReLU(inplace=True))
            input_dim = output_dim
            output_dim *= 2
        stem.append(nn.Conv2d(input_dim, embed_dim, kernel_size=1))
        self.proj = nn.Sequential(*stem)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x

def ctranspath():
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model

def scale_coordinate(data):
    """Convert imagecol and imagerow into high-resolution coordinates."""
    library_id = list(data.uns["spatial"].keys())[0]
    scale = data.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]

    if type(scale) != type(0.001):
        scale = float(scale)

    if type(data.obsm["spatial"][1,1]) == type('a'):
        data.obsm["spatial"] = data.obsm["spatial"].astype("float")
    image_coordinates = data.obsm["spatial"] * scale
    data.obs["imagecol"] = image_coordinates[:, 0]
    data.obs["imagerow"] = image_coordinates[:, 1]
    return data

def crop_images(data, crop_size=25):
    """Crop image based on crop_size and return an array of cropped images."""
    data = scale_coordinate(data)
    library_id = list(data.uns["spatial"].keys())[0]
    image_data = data.uns["spatial"][library_id]["images"]["hires"]
    # img = Image.fromarray(image_data)
    img = Image.fromarray((image_data * 255).astype(np.uint8))
    
    cropped_images = [
        img.crop((col - crop_size, row - crop_size, col + crop_size, row + crop_size))
        for row, col in zip(data.obs["imagerow"], data.obs["imagecol"])
    ]
    return np.stack([np.array(tile) / 255 for tile in cropped_images])

class ImageDataset(Dataset):
    def __init__(self, images_array, transform=None):
        self.images_array = images_array
        self.transform = transform

    def __len__(self):
        return self.images_array.shape[0]

    def __getitem__(self, idx):
        image = self.images_array[idx]
        if self.transform:
            # Convert numpy image to PIL image for transformation
            image = F.to_pil_image(image)
            image = self.transform(image)
        return image

    @staticmethod
    def collate_fn(batch):
        images = torch.stack(batch, dim=0)
        return images

def infer_by_pretrain(images, pretrain_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pre_model = ctranspath().to(device)
    pre_model.head = nn.Identity()
    pre_model.load_state_dict(torch.load(pretrain_path, map_location=device)['model'])
    pre_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    images_set = ImageDataset(images, transform=transform)
    dataloader = DataLoader(images_set, batch_size=32, shuffle=False, num_workers=4, collate_fn=ImageDataset.collate_fn)
    with torch.no_grad():
        features = [pre_model(batch.to(device)).cpu() for batch in dataloader]
    return torch.cat(features, dim=0)

def process_embeddings(embeddings, n_components, n_clusters):
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(pca_embeddings)
    return pca_embeddings, cluster_labels

def plot_patho_class_heatmap(data, save_path):
    """Plots a heatmap based on 'patho_class' labels and image coordinates."""
    # Extracting data
    x_coords = data.obs["array_col"].values
    y_coords = data.obs["array_row"].values

    if type(x_coords[0]) == type('a'):
        x_coords = x_coords.astype("int")
        y_coords = y_coords.astype("int")

    labels = data.obs["patho_class"].values
    
    # Number of unique labels (assuming they are sequential integers starting from 0)
    unique_labels = np.unique(labels)
    num_labels = len(unique_labels)
    
    # Define a list of distinct colors. Add more colors if you have more classes.
    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta', 'brown', 'pink', 'lime', 'violet', 'indigo', 'gold', 'crimson']
    if num_labels > len(colors):
        raise ValueError(f"Please define at least {num_labels} distinct colors.")
    
    # Create scatter plot with distinct colors
    plt.figure(figsize=(10, 10))
    for i, label in enumerate(unique_labels):
        mask = (labels == label)
        plt.scatter(x_coords[mask], y_coords[mask], color=colors[i], label=label, s=10)
    
    # Add legend, axis labels, and title
    plt.legend()
    plt.xlabel("Image Column")
    plt.ylabel("Image Row")
    plt.title("Patho Class Heatmap")
    
    # Display the plot
    plt.gca().invert_yaxis()  # Invert y-axis for typical image display
    plt.savefig(save_path,bbox_inches ="tight", pad_inches = 1)
    return None

def GetPathoClass(adata, pretrain_path, n_components = 50, n_clusters = 6, plot_classes = True, image_save_path = None):
    images = crop_images(adata)
    features = infer_by_pretrain(images, pretrain_path)

    # Example values for PCA and clustering
    pca_embeddings, cluster_labels = process_embeddings(features, n_components, n_clusters)
    adata.obs["patho_class"] = cluster_labels
    adata.obsm["patho_emd"] = pca_embeddings

    if plot_classes:
        plot_patho_class_heatmap(adata,image_save_path)

    return adata

