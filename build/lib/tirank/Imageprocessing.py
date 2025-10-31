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

"""
This module provides utilities for processing spatial transcriptomics (ST)
H&E (Hematoxylin and Eosin) images. It includes functions for:
1. Cropping image tiles (patches) centered on each ST spot.
2. Generating feature embeddings from these tiles using a pre-trained
   CTransPath (Swin Transformer) model.
3. Clustering these embeddings using PCA and K-means to identify
   spatial "pathological classes".
"""

def _ntuple(n):
    """
    Private helper function to create a tuple of size n.

    Args:
        n (int): The desired size of the tuple.

    Returns:
        function: A function that parses its input into a tuple of size n.
    """
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


class ConvStem(nn.Module):
    """
    Custom Convolutional Stem for the CTransPath model (replaces the patch embed layer).

    This stem uses a series of convolutions to create patch embeddings instead
    of a single large-kernel convolution.

    Args:
        img_size (int, optional): The size of the input image. Defaults to 224.
        patch_size (int, optional): The size of the patch. Must be 4. Defaults to 4.
        in_chans (int, optional): Number of input image channels. Defaults to 3.
        embed_dim (int, optional): The dimension of the output embedding. Defaults to 768.
        norm_layer (nn.Module, optional): Normalization layer to use. Defaults to None.
        flatten (bool, optional): Whether to flatten the output. Defaults to True.
    """

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
        """
        Forward pass of the convolutional stem.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Output patch embeddings.
        """
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


def ctranspath():
    """
    Factory function to create the CTransPath model.

    Initializes a Swin Transformer (swin_tiny_patch4_window7_224) with
    the custom ConvStem as the embedding layer.

    Returns:
        torch.nn.Module: The CTransPath model instance.
    """
    model = timm.create_model('swin_tiny_patch4_window7_224', embed_layer=ConvStem, pretrained=False)
    return model


def scale_coordinate(data):
    """
    Scales ST spot coordinates to match the high-resolution image.

    This function reads the scale factor from the AnnData object and applies
    it to the spatial coordinates, adding the results to `data.obs` as
    'imagecol' and 'imagerow'.

    Args:
        data (anndata.AnnData): The AnnData object containing spatial info.

    Returns:
        anndata.AnnData: The AnnData object, modified in place.
    """
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
    """
    Crops image tiles (patches) from the H&E slide for each spot.

    Uses the 'imagecol' and 'imagerow' coordinates from `data.obs` to
    crop square patches of (2*crop_size) x (2*crop_size) from the
    high-resolution image.

    Args:
        data (anndata.AnnData): The AnnData object, after running `scale_coordinate`.
        crop_size (int, optional): The "radius" for cropping. A size of 25
            creates 50x50 pixel tiles. Defaults to 25.

    Returns:
        np.ndarray: A NumPy array stack of image tiles of shape
            (n_spots, 2*crop_size, 2*crop_size, 3).
    """
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
    """
    PyTorch Dataset for loading image tiles (patches).

    Args:
        images_array (np.ndarray): A stack of image tiles from `crop_images`.
        transform (callable, optional): A torchvision transform to apply to each
            image. Defaults to None.
    """
    def __init__(self, images_array, transform=None):
        self.images_array = images_array
        self.transform = transform

    def __len__(self):
        """Returns the total number of images (spots)."""
        return self.images_array.shape[0]

    def __getitem__(self, idx):
        """
        Gets a single image and applies transformations.

        Args:
            idx (int): The index of the image to fetch.

        Returns:
            torch.Tensor: The transformed image tensor.
        """
        image = self.images_array[idx]
        if self.transform:
            # Convert numpy image to PIL image for transformation
            image = F.to_pil_image(image)
            image = self.transform(image)
        return image

    @staticmethod
    def collate_fn(batch):
        """
        Static collate function to stack images into a batch.

        Args:
            batch (list): A list of image tensors.

        Returns:
            torch.Tensor: A stacked batch of images.
        """
        images = torch.stack(batch, dim=0)
        return images


def infer_by_pretrain(images, pretrain_path):
    """
    Generates feature embeddings from image tiles using the pre-trained CTransPath.

    Args:
        images (np.ndarray): A stack of image tiles (n_spots, H, W, C).
        pretrain_path (str): Path to the CTransPath model's .pth weights file.

    Returns:
        torch.Tensor: A tensor of feature embeddings of shape (n_spots, n_features).
    """
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
    """
    Performs PCA dimensionality reduction and K-means clustering.

    Args:
        embeddings (torch.Tensor or np.ndarray): The feature embeddings.
        n_components (int): The number of principal components to keep.
        n_clusters (int): The number of clusters to find (k).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The PCA-transformed embeddings.
            - np.ndarray: The cluster labels for each spot.
    """
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.numpy()

    pca = PCA(n_components=n_components)
    pca_embeddings = pca.fit_transform(embeddings)
    cluster_labels = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(pca_embeddings)
    return pca_embeddings, cluster_labels


def plot_patho_class_heatmap(data, save_path):
    """
    Generates and saves a spatial scatter plot of pathological classes.

    Args:
        data (anndata.AnnData): The AnnData object with 'patho_class' in `.obs`.
        save_path (str): The file path to save the resulting plot.

    Returns:
        None
    """
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
        print(f"Warning: Not enough unique colors for {num_labels} clusters. Some colors will be repeated.")
        colors = colors * (num_labels // len(colors) + 1)
        
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
    """
    Orchestrates the full image processing pipeline.

    Crops tiles, generates embeddings using CTransPath, performs PCA and
    K-means, and adds the results to the AnnData object.

    Args:
        adata (anndata.AnnData): The AnnData object for an ST sample.
        pretrain_path (str): Path to the CTransPath model's .pth weights file.
        n_components (int, optional): Number of PCA components. Defaults to 50.
        n_clusters (int, optional): Number of K-means clusters. Defaults to 6.
        plot_classes (bool, optional): Whether to generate and save the spatial
            heatmap. Defaults to True.
        image_save_path (str, optional): File path to save the heatmap.
            Required if `plot_classes` is True. Defaults to None.

    Returns:
        anndata.AnnData: The AnnData object, modified in place with:
            - `adata.obs["patho_class"]`: Cluster labels for each spot.
            - `adata.obsm["patho_emd"]`: PCA embeddings for each spot.
    """
    images = crop_images(adata)
    features = infer_by_pretrain(images, pretrain_path)

    # Example values for PCA and clustering
    pca_embeddings, cluster_labels = process_embeddings(features, n_components, n_clusters)
    adata.obs["patho_class"] = cluster_labels
    adata.obsm["patho_emd"] = pca_embeddings

    if plot_classes:
        if image_save_path is None:
            raise ValueError("'image_save_path' must be provided if 'plot_classes' is True.")
        plot_patho_class_heatmap(adata,image_save_path)

    return adata