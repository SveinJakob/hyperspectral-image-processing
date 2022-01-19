# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 2021 13:26

@author: Svein Jakob Kristoffersen (sveinjakobkristoffersen@gmail.com)

Code to process the leaf data for my master thesis. Some code from Maria
Vukovic. This module consists of functions to be used for processing of data.
Bottom of file shows how to preprocess many images and save them to file.
"""

import os
from spectrum_corrections import *
from math import log10
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from spectral import envi
from spectral import kmeans, principal_components
from scipy.signal import savgol_filter
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, \
    recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

path = os.path.dirname(__file__)
os.chdir(path)


def read_image_file(img_path, hdr_path):
    """Reads and returns and image and hdr file"""
    # Read in the image and the headerfile
    img_read = envi.open(hdr_path, img_path).load()
    hdr_read = open(hdr_path, 'r').read()  # open the hdr file
    return img_read, hdr_read


def crop_to_area(img, area):
    """
    Crops image and returns cropped area.

    Will be used before k-means grouping to crop out the white reference for
    better groupig.

    ----- input -----
    img: spectral image
        Image to be cropped
    area: string
        Area of image to be cropped
    ----- returns -----
    img_cropped: spectral image
        The cropped image.
    """
    # Splitting area string into y- and x-ranges (strings)
    area_split = area.split(',')  # '250:270', '400:470'
    # Splittig the individual ranges into start and end values
    y_range_string = area_split[0].split(':')  # '250', '270'
    x_range_string = area_split[1].split(':')  # '400', '470'

    # Reaching start and end values and making slices
    y1, y2 = int(y_range_string[0]), int(y_range_string[1])
    x1, x2 = int(x_range_string[0]), int(x_range_string[1])
    y_range = slice(y1, y2)
    x_range = slice(x1, x2)

    img_cropped = img[y_range, x_range, :]
    return img_cropped


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


def crop_out_area_pixels(img, areas):
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
        areas_.append(area_pixels)

    # Return the list after all areas have been cropped:
    return areas_


def preprocess_image(dict, snv_corr=True):
    """
    Performs preprocessing on the image, erg. white reference correction,
    leaf cropping and leaf-background segmentation using k-means grouping.

    ----- input -----
    dict: dictionary
        Contains path to image file, area of white ref, area of leaf-crop and
        possibly area of rust.
    ----- returns -----
    leaf_pixels: list
        List containing all the pixels from the leaf
    mask: matrix
        matrix contaning the mask for leaf-background segmentation
    c: array(?)
        centers found with k-means
    leaf: spectral image
        The white reference corrected and segmented leaf image.
    rust_pixels: list
        List containing all the rust pixels from the leaf (option)
    """
    path_img = dict['Path image']
    path_hdr = dict['Path hdr']
    ref_area = dict['Area w.r.']
    leaf_area = dict['Area leaf']
    if 'Areas rust' in dict.keys():  # If we have a rust area
        rust_areas = dict['Areas rust']
    if 'Areas healthy' in dict.keys():  # If we have a healthy area
        healthy_areas = dict['Areas healthy']
    if 'Areas senescence' in dict.keys():  # If we have a senescence area
        senescence_areas = dict['Areas senescence']

    # Read and WR. corr. the image:
    img, hdr_file = read_image_file(path_img, path_hdr)
    img = white_reference_correction(img, ref_area)
    # Crop out leaf area:
    leaf_img = crop_to_area(img, leaf_area)
    # Group the leaf pixels:
    leaf_pixels, mask, c = group_with_kmeans(leaf_img)

    # If areas, return with the cropped out pixels:
    if 'Areas rust' in dict.keys():
        area_pixels = crop_out_area_pixels(img, rust_areas)
    elif 'Areas healthy' in dict.keys():
        area_pixels = crop_out_area_pixels(img, healthy_areas)
    elif 'Areas senescence' in dict.keys():
        area_pixels = crop_out_area_pixels(img, senescence_areas)
    else:
        area_pixels = []

    # If snv is wanted:
    if snv_corr is True:
        leaf_pixels = snv(leaf_pixels)
        area_pixels = snv_on_list(area_pixels)

    return leaf_pixels, mask, leaf_img, area_pixels


def preprocess_images_in_dict(dict_of_dicts, snv_corr=True):
    """
    Functions which calls function preprocess_image on all images in given dict
    """
    # Preprocess all images and append them to lists --------------------------
    all_leaf_pixels = []
    all_leaf_images = []
    all_area_pixels = []
    i = 0
    for dict_ in dict_of_dicts:
        i += 1
        print(f'Processing image:  {i}')
        leaf_pixels, mask, leaf_img, area_pixels = preprocess_image(dict_,
                                                            snv_corr=snv_corr)
        all_leaf_pixels.append(leaf_pixels)
        all_leaf_images.append(leaf_img)
        all_area_pixels.append(area_pixels)
    return all_leaf_pixels, all_leaf_images, all_area_pixels


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


def classifier_img_pred(X, y, img, mask, pca=None, classifier=None,
                        sensitive_bands=None, n_classes=2, emsc_corr=False,
                        emsc_degree=0, emsc_Xref=None):
    """
    Function to classify pixels on passed hyperspectral image with passed
    classifier trained on passed data.

    Parameters
    ----------
    X: array
        Matrix containing spectrums
    y: array
        Array containing classes of spectrums in X
    img: array
        3D matrix (hypercube) containing spectrums to be classified
    mask: array
        2D binary matrix masking leaf on img
    pca: class
        PCA from sklearn to use if PCA transformation needed
    classifier: class
        Sklearn class to classify spectrums from img
    sensitive_bands: array
        Bands to crop img if necessary
    n_classes: int
        2 or 4 classes (method 1 or 2)
    emsc_corr: bool
        If emsc is necessary
    emsc_degree: int
        Degree of emsc
    emsc_Xref: array
        Reference spectrum to use in emsc

    Returns
    -------
    classified_img: array
        2D matrix of img classified

    """
    classifier.fit(X, y)
    # Using this classifyer to classify pixels on image:
    img_to_classify = img
    img_classified = np.zeros(shape=(img_to_classify.shape[0],
                                     img_to_classify.shape[1]))
    for i in range(img_to_classify.shape[0]):
        for j in range(img_to_classify.shape[1]):
            if mask[i,j] == 0:  # If this pixel is background
                img_classified[i,j] = -1
            else:  # If this pixel is leaf
                spectrum = img_to_classify[i, j, :].reshape(-1, 1).T

                if pca is not None:
                    spectrum = pca.transform(spectrum)
                if emsc_corr is True:
                    spectrum = emsc(X=spectrum, ref_spec=emsc_Xref,
                                    d=emsc_degree)
                if sensitive_bands is not None:
                    spectrum = spectrum[:, sensitive_bands]

                pred = classifier.predict(spectrum)
                if n_classes == 2:
                    if pred < 0.5:
                        img_classified[i, j] = 0
                    elif 0.5 < pred:
                        img_classified[i, j] = 1
                elif n_classes == 4:
                    if -0.5 < pred < 0.5:
                        img_classified[i, j] = 0
                    elif 0.5 < pred < 1.5:
                        img_classified[i, j] = 1
                    elif 1.5 < pred < 2.5:
                        img_classified[i, j] = 2
                    elif 2.5 < pred < 3.5:
                        img_classified[i, j] = 3
    return img_classified


# -----------------------------------------------------------------------------

def preprocess_leaf_img(dict, snv_corr=True):
    """
    Performs preprocessing on the image, erg. white reference correction,
    leaf cropping and leaf-background segmentation using k-means grouping and
    and optional SNV correction.

    ----- input -----
    dict: dictionary
        Contains path to image file, area of white ref, area of leaf-crop and
        possibly area of choosing.
    ----- returns -----
    leaf_pixels: list
        List containing all the pixels from the leaf
    mask: matrix
        matrix contaning the mask for leaf-background segmentation
    leaf_img: spectral image
        The white reference corrected and segmented leaf image.
    area_pixels: list
        List containing all the area pixels from the leaf (option)
    """
    path_img = dict['Path image']
    path_hdr = dict['Path hdr']
    ref_area = dict['Area w.r.']
    leaf_area = dict['Area leaf']
    areas_of_interest = dict['AOI']

    # Read and WR. corr. the image:
    img, hdr_file = read_image_file(path_img, path_hdr)
    img = white_reference_correction(img, ref_area)
    # Crop out leaf area:
    leaf_img = crop_to_area(img, leaf_area)
    # Group the leaf pixels:
    print('Starting Kmeans')
    leaf_pixels, mask, c = group_with_kmeans(leaf_img)
    print('Kmeans ended')

    # If areas, return with the cropped out pixels:
    print('Cropping out AOI')
    area_pixels = get_areas_from_img(img, areas_of_interest)
    print(f'AOI cropped: shape{np.shape(area_pixels)}')

    # If snv is wanted:
    if snv_corr is True:
        print('SNV correction applying')
        leaf_pixels = snv(leaf_pixels)
        area_pixels = snv(area_pixels)
        leaf_img = snv_on_image(leaf_img)
        print('SNV correction applied')

    return leaf_pixels, mask, leaf_img, area_pixels


def read_data_to_df(dicts, snv_corr=True):
    """
    Reads all data from input, preprocesses it and puts it into a dataframe.

    Extracts each dictionary from dicts and preprocesses its data with external
    function preprocess_leaf_img(). All preprosessed leaf data is placed in a
    dataframe with column names:
        class: str
            ex. rust, healthy
        leaf pixels: array
            array of arrays representing pixel spectrums from leaf.
        area pixels: array
            contains arrays representing spectra of pixels in specified areas
        leaf number: int
            number to keep track of leaf for RGB comparison.
        image: matrix
            hypercube of leaf image.
        mask: matrix
            mask for leaf with binary values.

    This function could be improved to also be able to perform EMSC on spectra.

    ----- input -----
    dicts: dictionary
        Dictionary containing many dictionaries, one for each leaf.
    snv_corr: bool
        Whether or not to perform SNV correction on spectra.
    ----- returns -----
    df: DataFrame
        Contains columns as described above, each row represent data for one
        leaf.
    """
    df = pd.DataFrame()
    for dict_ in dicts:
        leaf_pixels, mask, img, area_pixels = \
            preprocess_leaf_img(dict_, snv_corr=snv_corr)
        df_leaf = pd.DataFrame(columns=['class', 'leaf pixels', 'area pixels',
                                        'leaf number', 'image', 'mask'])
        df_leaf['class'] = [dict_['Class']]
        df_leaf['leaf pixels'] = [leaf_pixels]
        df_leaf['area pixels'] = [area_pixels]
        df_leaf['leaf number'] = [dict_['Path image'].split('\\')[-1][:-4]]
        df_leaf['image'] = [img]
        df_leaf['mask'] = [mask]
        df = df.append(df_leaf)
    return df


if __name__ == '__main__':

    df = read_data_to_df(leaves_160720, snv_corr=False)
    df.to_pickle('leaf_dataframe_train16.csv', protocol=4)
    print('Done!')
