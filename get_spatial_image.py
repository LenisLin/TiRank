import os
import sys
import numpy as np
import anndata
import scanpy as sc
import pandas as pd
from anndata import AnnData
from PIL import Image

#示例数据位置
path="/mnt/data/songjinsheng/a_graduate/dataset/ST_CRC/SN048_A121573_Rep1"

#读取数据
adata = sc.read_visium(path, 
                    count_file = 'filtered_feature_bc_matrix.h5',
                    load_images = True,
                    )

#将imagecol和imagerow均转化为highres中坐标而不是原先的fullres坐标
def scale_coordinate(adata):
    library_id = list(adata.uns["spatial"].keys())[0]
    scale = adata.uns["spatial"][library_id]["scalefactors"]["tissue_hires_scalef"]
    image_coor = adata.obsm["spatial"] * scale
    adata.obs["imagecol"] = image_coor[:, 0]
    adata.obs["imagerow"] = image_coor[:, 1]
    return(adata)

#剪切图像，crop_size代表图片半径（边长的1/2），最终返回一个list，list中每个元素是一个[2*crop_size,2*crop_size,3]的array
def image_crop(
        adata,
        crop_size=16
        ):
    #crop_size=int(adata.uns["spatial"][library_id]["scalefactors"]["spot_diameter_fullres"]*scale)   
    library_id = list(adata.uns["spatial"].keys())[0]

    image = adata.uns["spatial"][library_id]["images"]["hires"]
    image = (image * 255).astype(np.uint8)
    img_pillow = Image.fromarray(image)
    img_array_list = []

    for imagerow, imagecol in zip(adata.obs["imagerow"], adata.obs["imagecol"]):
        imagerow_down = imagerow - crop_size
        imagerow_up = imagerow + crop_size
        imagecol_left = imagecol - crop_size
        imagecol_right = imagecol + crop_size
        tile = img_pillow.crop((imagecol_left, imagerow_down, imagecol_right, imagerow_up))
        img_array = np.array(tile)
        img_array_list.append(img_array)

    return img_array_list


#运行
adata=scale_coordinate(adata)
img_array_list=image_crop(adata)
