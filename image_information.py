# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 2022 19:00

@author: Svein Jakob Kristoffersen (sveinjakobkristoffersen@gmail.com)

Functions used to get information from images.

"""

import numpy as np
from spectral import kmeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from spectral import kmeans




def plot_image_

def quickplot_image(path_img, path_hdr, area_wr=[0,0,0,0], area_leaf=[0,0,0,0],
                    bands=[0, 288],
                    clim=(0, 0.02)):
    """Simple image plot to check areas"""
    img, hdr = read_image_file(path_img, path_hdr)
    # plt.rc('font', size=8)
    dpi = 100
    plt.figure(figsize=(img.shape[1]/dpi, img.shape[0]/dpi), dpi=dpi)
    # Testing better visualization:
    plt.imshow(np.mean(img[:, :, bands[0]:bands[1]], axis=2),
               cmap='jet', clim=clim)
    plt.colorbar()

    # Drawing the white reference and leaf area
    ymin_wr = area_wr[0]; ymax_wr = area_wr[1]
    xmin_wr = area_wr[2]; xmax_wr = area_wr[3]
    ymin_l = area_leaf[0]; ymax_l = area_leaf[1]
    xmin_l = area_leaf[2]; xmax_l = area_leaf[3]

    # # White reference area:
    plt.gca().add_patch(Rectangle((xmin_wr, ymin_wr),
                                  (xmax_wr - xmin_wr), (ymax_wr - ymin_wr),
                                  linewidth=1, color='r', fill=False))
    # # Leaf area:
    plt.gca().add_patch(Rectangle((xmin_l, ymin_l),
                                  (xmax_l - xmin_l), (ymax_l - ymin_l),
                                  linewidth=1, color='r', fill=False))
    plt.show()
    return img


def plot_hypercube(cube, mask, bands, Xref_emsc=False, emsc_degree=2,
                   clim=(0,1)):
    """Visualizing the hypercube, applying EMSC if necessary"""
    for i in range(cube.shape[0]):
        for j in range(cube.shape[1]):
            if mask[i,j] == 0:  # If this pixel is background
                cube[i,j] = 0
            elif Xref_emsc is not None:
                cube[i, j, :], = emsc(X=cube[i, j, :].reshape(-1, 1).T,
                                      ref_spec=Xref_emsc, d=emsc_degree)
    cube = cube[:, :, bands]
    plt.figure(figsize=(5, 12))
    plt.imshow(cube, clim=clim, interpolation='None')
    plt.colorbar()
    plt.show()


def group_with_kmeans(img):
    """
    Groups the pixels in image in two groups for leaf-background segmentation.

    To be used on cropped image containing leaf and background

    ----- input -----
    img: spectral image
        Image containing leaf and background to be segmented
    ----- returns -----
    leaf_pixels: list
        List containing all the pixels from the leaf
    mask: matrix
        matrix contaning the mask for leaf-background segmentation
    c: array(?)
        centers found with kmeans.
    """
    (mask, c) = kmeans(img, 2, 20)

    leaf_pixels = []
    for i in range(np.shape(mask)[0]):
        for j in range(np.shape(mask)[1]):
            if mask[i, j] == 1:
                leaf_pixels.append(img[i, j, :])
    leaf_pixels = np.array(leaf_pixels)
    return leaf_pixels, mask, c


def give_deviation_graphs(list_of_spectrums):
    """Returns mean and std of list of spectrums"""
    std = np.std(list_of_spectrums, axis=0)
    mean_spectrum = np.mean(list_of_spectrums, axis=0)
    return mean_spectrum + std, mean_spectrum - std


def reach_area_pixels(df_, class_, is_healthy=False):
    """Returns wanted areas from dataframe, used for abstraction"""
    if is_healthy:
        area_pixels_ = df_.loc[df_['class'] == class_, 'leaf pixels']
        area_pixels = []
        for area in area_pixels_:
            area_pixels.extend(area)
    else:
        area_pixels_ = df_.loc[df_['class'] == class_, 'area pixels']
        area_pixels = []
        for area in area_pixels_:
            area_pixels.extend(area)
    return np.array(area_pixels)


def get_wavelength(keep_floats=False):
    """
    Returns the wavelengths used for one hdr-file, common wavelengths for
    all images
    """
    hdr = r'C:\Users\svein\Desktop\Rust\leaves160720\rust_01.hdr'
    hdr_file = open(hdr, 'r').read()  # open the hdr file
    wavelength = hdr_file[148:2989 + 7].split(',')
    if keep_floats:
        return np.array([float(wavelength[i])
                         for i in range(len(wavelength))])
    else:
        return np.array([int(float(wavelength[i]))
                         for i in range(len(wavelength))])


def get_areas_from_img(img, areas):
    """
    Function to crop out the chosen areas.

    ----- input -----
    img: spectral image
        Image containing the rust pixels to crop out
    area: string
        Areas in image to crop out, format [ymin, ymax, xmin, xmax]
    ----- returns -----
    areas_: list
        Contains list of all areas' pixels from the image.
    """
    areas_ = []
    for area in areas:
        area_pixels = []
        ymin = area[0]
        ymax = area[1]
        xmin = area[2]
        xmax = area[3]

        # Append the crop to list:
        img_area = img[ymin:ymax, xmin:xmax, :]
        for i in range(img_area.shape[0]):
            for j in range(img_area.shape[1]):
                area_pixels.append(img_area[i, j, :])
        areas_.extend(area_pixels)

    # Return the list after all areas have been cropped:
    return np.array(areas_)
