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
import numpy as np
from spectral import envi
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, \
    recall_score, precision_score
from sklearn.model_selection import StratifiedKFold

from image_corrections import *
from image_cropping import *
from image_information import *

path = os.path.dirname(__file__)
os.chdir(path)


def read_image_file(img_path, hdr_path):
    """Reads and returns and image and hdr file"""
    # Read in the image and the headerfile
    img_read = envi.open(hdr_path, img_path).load()
    hdr_read = open(hdr_path, 'r').read()  # open the hdr file
    return img_read, hdr_read


def preprocess_image(_image_info, emsc_info=None):
    """
    Performs simple preprocessing of image.

    White reference correction as well as optimal EMSC. The mask of the image
    is also created using K-means grouping.

    ----- input -----
    _image_info: dict
        Contains path to image and hdr file, area of white ref, area of leaf.
    emsc_info: dict
        (Optional) If passed, EMSC correction is performed. Should contain ref-
        spectrum and degree to perform correction.
    ----- returns -----
    img: np.array
        image as 3-dimensional matrix
    mask: np.array
        mask of leaf, 2-dimensional matrix with bg=0 and leaf=1
    """
    # Getting information:
    path_img = _image_info['Path image']
    path_hdr = _image_info['Path hdr']
    ref_area = _image_info['Area w.r.']
    leaf_area = _image_info['Area leaf']

    # Read and WR. corr. the image:
    img, hdr_file = read_image_file(path_img, path_hdr)
    img = image_correction_white_reference(img, ref_area)

    # Crop out leaf area:
    img = crop_to_area(img, leaf_area)

    # Group the leaf pixels:
    _, mask, _ = group_with_kmeans(img)

    if emsc_info is not None:
        ref_spectrum = emsc_info['ref_spectrum']
        degree = emsc_info['degree']
        img = image_correction_emsc(img, mask, ref_spectrum, degree)

    return img, mask


# def preprocess_images_in_dict(dict_of_dicts, snv_corr=True):
#     """
#     Functions which calls function preprocess_image on all images in given dict
#     """
#     # Preprocess all images and append them to lists --------------------------
#     all_leaf_pixels = []
#     all_leaf_images = []
#     all_area_pixels = []
#     i = 0
#     for dict_ in dict_of_dicts:
#         i += 1
#         print(f'Processing image:  {i}')
#         leaf_pixels, mask, leaf_img, area_pixels = preprocess_image(dict_,
#                                                             snv_corr=snv_corr)
#         all_leaf_pixels.append(leaf_pixels)
#         all_leaf_images.append(leaf_img)
#         all_area_pixels.append(area_pixels)
#     return all_leaf_pixels, all_leaf_images, all_area_pixels


def predict_image_with_model(img, mask, classifier=None, n_classes=2,
                             emsc_info=None):
    """
    Function to classify pixels on hyper spectral image with passed classifier.

    Parameters
    ----------
    img: np.array
        3D matrix (hypercube) containing spectrums to be classified
    mask: np.array
        2D binary matrix masking leaf on img
    classifier: class
        Trained classifier with a .predict() method.
    n_classes: int
        2 or 4 classes
    emsc_info: dict
        passed if EMSC wanted, should contain degree and reference spectrum

    Returns
    -------
    classified_img: array
        2D matrix of img classified

    """
    # Array with same dimensions as img, classified values are put into this:
    img_classified = np.zeros(shape=(img.shape[0], img.shape[1]))

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):

            if mask[i, j] == 0:  # If this pixel is background
                img_classified[i, j] = -1  # Setting background as value -1

            else:  # If this pixel is leaf

                spectrum = img[i, j, :].reshape(-1, 1).T  # The spectrum

                # EMSC if emsc_info is given:
                if emsc_info is not None:
                    spectrum = emsc(X=spectrum,
                                    ref_spec=emsc_info['ref_spectrum'],
                                    d=emsc_info['degree'])

                # Making a prediction with the model:
                pred = classifier.predict(spectrum)

                # Using the prediction to append integers to img_classified.
                # This might change dependent on classifier output.
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


# def preprocess_leaf_img(dict, snv_corr=True):
#     """
#     Performs preprocessing on the image, erg. white reference correction,
#     leaf cropping and leaf-background segmentation using k-means grouping and
#     and optional SNV correction.
#
#     ----- input -----
#     dict: dictionary
#         Contains path to image file, area of white ref, area of leaf-crop and
#         possibly area of choosing.
#     ----- returns -----
#     leaf_pixels: list
#         List containing all the pixels from the leaf
#     mask: matrix
#         matrix contaning the mask for leaf-background segmentation
#     leaf_img: spectral image
#         The white reference corrected and segmented leaf image.
#     area_pixels: list
#         List containing all the area pixels from the leaf (option)
#     """
#     path_img = dict['Path image']
#     path_hdr = dict['Path hdr']
#     ref_area = dict['Area w.r.']
#     leaf_area = dict['Area leaf']
#     areas_of_interest = dict['AOI']
#
#     # Read and WR. corr. the image:
#     img, hdr_file = read_image_file(path_img, path_hdr)
#     img = white_reference_correction(img, ref_area)
#     # Crop out leaf area:
#     leaf_img = crop_to_area(img, leaf_area)
#     # Group the leaf pixels:
#     print('Starting Kmeans')
#     leaf_pixels, mask, c = group_with_kmeans(leaf_img)
#     print('Kmeans ended')
#
#     # If areas, return with the cropped out pixels:
#     print('Cropping out AOI')
#     area_pixels = get_areas_from_img(img, areas_of_interest)
#     print(f'AOI cropped: shape{np.shape(area_pixels)}')
#
#     # If snv is wanted:
#     if snv_corr is True:
#         print('SNV correction applying')
#         leaf_pixels = snv(leaf_pixels)
#         area_pixels = snv(area_pixels)
#         leaf_img = snv_on_image(leaf_img)
#         print('SNV correction applied')
#
#     return leaf_pixels, mask, leaf_img, area_pixels


# def read_data_to_df(dicts, snv_corr=True):
#     """
#     Reads all data from input, preprocesses it and puts it into a dataframe.
#
#     Extracts each dictionary from dicts and preprocesses its data with external
#     function preprocess_leaf_img(). All preprosessed leaf data is placed in a
#     dataframe with column names:
#         class: str
#             ex. rust, healthy
#         leaf pixels: array
#             array of arrays representing pixel spectrums from leaf.
#         area pixels: array
#             contains arrays representing spectra of pixels in specified areas
#         leaf number: int
#             number to keep track of leaf for RGB comparison.
#         image: matrix
#             hypercube of leaf image.
#         mask: matrix
#             mask for leaf with binary values.
#
#     This function could be improved to also be able to perform EMSC on spectra.
#
#     ----- input -----
#     dicts: dictionary
#         Dictionary containing many dictionaries, one for each leaf.
#     snv_corr: bool
#         Whether or not to perform SNV correction on spectra.
#     ----- returns -----
#     df: DataFrame
#         Contains columns as described above, each row represent data for one
#         leaf.
#     """
#     df = pd.DataFrame()
#     for dict_ in dicts:
#         leaf_pixels, mask, img, area_pixels = \
#             preprocess_leaf_img(dict_, snv_corr=snv_corr)
#         df_leaf = pd.DataFrame(columns=['class', 'leaf pixels', 'area pixels',
#                                         'leaf number', 'image', 'mask'])
#         df_leaf['class'] = [dict_['Class']]
#         df_leaf['leaf pixels'] = [leaf_pixels]
#         df_leaf['area pixels'] = [area_pixels]
#         df_leaf['leaf number'] = [dict_['Path image'].split('\\')[-1][:-4]]
#         df_leaf['image'] = [img]
#         df_leaf['mask'] = [mask]
#         df = df.append(df_leaf)
#     return df


if __name__ == '__main__':

    df = read_data_to_df(leaves_160720, snv_corr=False)
    df.to_pickle('leaf_dataframe_train16.csv', protocol=4)
    print('Done!')
