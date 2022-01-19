# -*- coding: utf-8 -*-
"""
Created on Wed Jan 13 2021 13:26

@author: Svein Jakob Kristoffersen (sveinjakobkristoffersen@gmail.com)

Code to process the leaf data for my master thesis. Some code from Maria
Vukovic. This module consists of functions to be used for processing of data.
Bottom of file shows how to preprocess many images and save them to file.
"""

import os
from emsc import emsc
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


def white_reference_correction(img, area):
    """
    Perform correction with the white reference, given its area in image.

    ---- input -----
    img: spectral image
        Image with a white reference area in it
    area: string
        Area of the white reference in the image. Ex. '250:270, 400:470'
    ---- returns -----
    img_corr: spectral image
        The corrected image.
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

    # Slicing the area containing the white reference
    white_ref = img[y_range, x_range, :]
    # Median of the white reference
    white_ref_median = np.median(white_ref, axis=(0, 1))

    # Carry through white reference correction
    img_corr = img / white_ref_median
    return img_corr


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

def snv(spectrum_list):
    """
    Performs standard normal variate correction on spectrums in a list.

    https://nirpyresearch.com/two-scatter-correction-techniques-nir-spectroscopy-python/
    """
    output_spectrum = np.zeros_like(spectrum_list)
    spectrum_list = np.array(spectrum_list)
    for i in range(spectrum_list.shape[0]):
        output_spectrum[i, :] = (spectrum_list[i,:] -
                                 np.mean(spectrum_list[i,:])) \
                                / np.std(spectrum_list[i,:])
    return output_spectrum


def snv_on_list(list_of_pixels):
    """
    Perform standard normal variate correction on a list of spectrums

    The only difference of this and the above function is the shape of input,
    more specifically the depth of the list erg. [[], []] vs [[ [], [] ]]
    """
    list_of_pixels_corr = []
    # for first_bracket in list_of_pixels:
    for pixel_list in list_of_pixels:
        list_of_pixels_corr.append(snv(pixel_list))
    return list_of_pixels_corr

def snv_single_spectrum(spectrum):
    """Perform standard normal variate correction on a single spectrum"""
    return (spectrum - np.mean(spectrum)) / np.std(spectrum)

def snv_on_image(img):
    """
    Performs standard normal variate correction correction on an entire image

    """
    img_corr = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_corr[i, j, :] = snv_single_spectrum(img[i, j, :])
    return np.array(img_corr)


def _absorbance_value(value):
    """Function to convert input value (reflectance) to absorbance."""
    assert value > 0  # if value is less than zero, log10 not possible
    return log10(1/value)

def absorbance_spectrum(spectrum):
    """Function to convert input spectrum (reflectance) to absorbance."""
    assert spectrum.shape == (1, 288) or spectrum.shape == (288,), \
        f'Incorrect shape of input spectrum: {spectrum.shape}'
    new_spectrum = np.zeros_like(spectrum)
    for i, value in enumerate(spectrum):
        new_spectrum[i] = _absorbance_value(value)
    return new_spectrum


def absorbance_matrix(datamatrix):
    """
    Functino to convert matrix of spectra (reflectance) to matrix of
    absorbance spectra.

    """
    new_matrix = np.zeros_like(datamatrix)
    for i, spectrum in enumerate(datamatrix):
        new_matrix[i, :] = absorbance_spectrum(spectrum)
    return new_matrix


def _aquagram_single_value(value, mean, std):
    """Applies formula for aquagram value on a single value"""
    return (value - mean)/std


def aquagram_values(matrix1, matrix2, matrix3, matrix4):
    """Function to turn all values in given bands into aquagram values"""
    m1_range = range(0, matrix1.shape[0])
    m2_range = range(m1_range[-1] + 1, m1_range[-1] + matrix2.shape[0] + 1)
    m3_range = range(m2_range[-1] + 1, m2_range[-1] + matrix3.shape[0] + 1)
    m4_range = range(m3_range[-1] + 1, m3_range[-1] + matrix4.shape[0] + 1)

    X = np.concatenate([matrix1, matrix2, matrix3, matrix4])
    X_new = np.zeros_like(X)
    for column in range(X.shape[1]):
        mean = np.mean(X[:, column])
        std = np.std(X[:, column])
        for row in range(X.shape[0]):
            X_new[row, column] = _aquagram_single_value(X[row, column],
                                                        mean, std)
    return X_new[m1_range, :], X_new[m2_range, :], X_new[m3_range, :], \
           X_new[m4_range, :]


def pca(image):
    """Performs PCA on an entire image using the spectral package"""
    pc = principal_components(image)
    pc_0999 = pc.reduce(fraction=0.999)
    loading = pc_0999.eigenvectors
    score = pc_0999.transform(image)
    return loading, score


def _first_derivative_of_list(list_with_spectrums, savgol=True,
                              window_length=7, polyorder=5):
    """
    Returns the first derivative of all spectrums in passed list. Assumes dx=1.
    """
    first_derivative_spectrums = []
    for spectrum in list_with_spectrums:
        first_der = np.diff(spectrum)
        if savgol:
            first_der = savgol_filter(first_der, window_length, polyorder)
        first_derivative_spectrums.append(first_der)
    return np.array(first_derivative_spectrums)


def _second_derivative_of_list(list_with_spectrums, savgol=True,
                               window_length=7, polyorder=5):
    """
    Returns the seoncd derivative of all spectrums in passed list. Assumes
    dx=1.
    """
    second_derivative_spectrums = []
    for spectrum in list_with_spectrums:
        first_der = np.diff(spectrum)
        second_der = np.diff(first_der)
        if savgol:
            second_der = savgol_filter(second_der, window_length, polyorder)
        second_derivative_spectrums.append(second_der)
    return np.array(second_derivative_spectrums)


def make_first_and_second_der(pixel_list):
    """
    Takes a list of spectrums as input and returns them as first and second
    derivatives in two lists.
    """
    first_der = _first_derivative_of_list(pixel_list, savgol=True,
                                          window_length=7, polyorder=5)
    # Arrays with second derivative of spectrums
    second_der = _second_derivative_of_list(pixel_list, savgol=True,
                                            window_length=11, polyorder=3)
    return first_der, second_der


def make_x_y(list_of_lists):
    """
    Makes X and y matrices of list of lists for use in machine learning.
    Each list in list of lists correspond to a class.
    """
    y = []
    X = []

    i = 0
    for single_list in list_of_lists:
        y.extend(i for _ in range(len(single_list)))
        i += 1
        X.extend(single_list)
    return np.array(X), np.array(y)

def give_deviation_graphs(list_of_spectrums):
    """Returns mean and std of list of spectrums"""
    std = np.std(list_of_spectrums, axis=0)
    mean_spectrum = np.mean(list_of_spectrums, axis=0)
    return mean_spectrum + std, mean_spectrum - std

class PLSDA_crossval():
    """Dummyclass used to replicate the behaviour of a real classifier."""
    def __init__(self, n_components, n_classes=2):
        self.n_components = n_components
        self.plsr = PLSRegression(n_components=n_components)

        self.y_true = None
        self.y_pred = None
        assert n_classes in [2, 4] # Function can only handle 2 or 4 classes
        self.n_classes = n_classes

    def f1(self):
        return f1_score(y_true=self.y_true, y_pred=self.y_pred,
                        average='weighted')

    def precision(self):
        return precision_score(y_true=self.y_true, y_pred=self.y_pred,
                               average='weighted')

    def recall(self):
        return recall_score(y_true=self.y_true, y_pred=self.y_pred,
                            average='weighted')

    def confm(self):
        return confusion_matrix(y_true=self.y_true, y_pred=self.y_pred)

    def statistics(self, y_true=None, y_pred=None):
        """Returns various statistics"""
        self.y_true = y_true
        self.y_pred = y_pred
        return self.f1(), self.precision(), self.recall(), self.confm()

    def cross_val(self, cv=5, X=None, y_true=None):
        """Performs crossvalidation on input data and returns statistics"""
        self.plsr = PLSRegression(n_components=self.n_components)
        strat_kfold = StratifiedKFold(n_splits=cv)
        outer_acc = []
        outer_f1 = []
        outer_rec = []
        outer_pre = []
        if self.n_classes == 2:
            outer_confm = np.zeros(shape=(2, 2))
        elif self.n_classes == 4:
            outer_confm = np.zeros(shape=(4, 4))

        for train_i, test_i in strat_kfold.split(X, y_true):
            self.plsr.fit(X[train_i], y_true[train_i])
            y_pred = self.plsr.predict(X[test_i])
            if self.n_classes == 2:
                y_pred = (y_pred > 0.5).astype('uint8')  # Making the pred 1/0
            elif self.n_classes == 4:
                y_pred = np.around(y_pred).astype(int)
                y_pred = np.clip(y_pred, 0, 3)

            outer_acc.append(accuracy_score(y_true[test_i], y_pred))

            f1, prec, reca, confm = self.statistics(y_true=y_true[test_i],
                                                    y_pred=y_pred)

            outer_f1.append(f1)
            outer_pre.append(prec)
            outer_rec.append(reca)
            outer_confm = outer_confm + confm
            print(confm)

        return np.mean(outer_acc), np.std(outer_acc), np.mean(outer_f1), \
               np.mean(outer_pre), np.mean(outer_rec), outer_confm


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


def undersample_list(majority_list, new_size=None):
    """
    Performs random removal of elements in a list, until it reaches wanted
    size
    """
    indices = np.random.choice(majority_list.shape[0], new_size)
    return majority_list[indices]


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


# -----------------------------------------------------------------------------
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
    leaves_160720 = [
        {
            'Path image': path_folder160720 + r'\rust_02.img',
            'Path hdr': path_folder160720 + r'\rust_02.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:560, 50:130',
            'Class': 'rust',
            'AOI': [[448,450,88,90], [452,458,89,90]]},
        {
            'Path image': path_folder160720 + r'\rust_03.img',
            'Path hdr': path_folder160720 + r'\rust_03.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '110:500, 50:130',
            'Class': 'rust',
            'AOI': [[274, 280, 101, 102], [302, 304, 101, 102], [312, 314, 107,
                                                                 108]]},
        {
            'Path image': path_folder160720 + r'\rust_06.img',
            'Path hdr': path_folder160720 + r'\rust_06.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '180:550, 70:160',
            'Class': 'rust',
            'AOI': [[307, 310, 109, 110]]},
        {
            'Path image': path_folder160720 + r'\rust_07.img',
            'Path hdr': path_folder160720 + r'\rust_07.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '140:560, 50:130',
            'Class': 'rust',
            'AOI': [[386, 388, 90, 91]]},
        {
            'Path image': path_folder160720 + r'\rust_08.img',
            'Path hdr': path_folder160720 + r'\rust_08.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '140:570, 50:140',
            'Class': 'rust',
            'AOI': [[320, 330, 99, 101]]},
        {
            'Path image': path_folder160720 + r'\rust_10.img',
            'Path hdr': path_folder160720 + r'\rust_10.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '150:480, 50:140',
            'Class': 'rust',
            'AOI': [[314, 319, 105, 107]]},
        {
            'Path image': path_folder160720 + r'\rust_11.img',
            'Path hdr': path_folder160720 + r'\rust_11.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '150:500, 50:140',
            'Class': 'rust',
            'AOI': [[322, 323, 102, 104], [331, 334, 102, 103]]},
        {
            'Path image': path_folder160720 + r'\rust_12.img',
            'Path hdr': path_folder160720 + r'\rust_12.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '100:480, 50:140',
            'Class': 'rust',
            'AOI': [[254, 257, 99, 102]]},
        {
            'Path image': path_folder160720 + r'\rust_13.img',
            'Path hdr': path_folder160720 + r'\rust_13.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:550, 50:140',
            'Class': 'rust',
            'AOI': [[486, 489, 81, 83], [440, 449, 84, 84], [435, 436, 84, 85],
                    [217, 223, 98, 99]]},

        # blotch necrosis
        {
            'Path image': path_folder160720 + r'\blotch_01.img',
            'Path hdr': path_folder160720 + r'\blotch_01.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:550, 50:140',
            'Class': 'blotch necrosis',
            'AOI': [[289, 294, 91, 92], [457, 467, 90, 92], [484, 489, 86,
                                                             87]]},
        {
            'Path image': path_folder160720 + r'\blotch_02.img',
            'Path hdr': path_folder160720 + r'\blotch_02.hdr',
            'Area w.r.': '70:380, 220:350',
            'Area leaf': '120:670, 50:140',
            'Class': 'blotch necrosis',
            'AOI': [[472, 475, 105, 107]]},
        {
            'Path image': path_folder160720 + r'\blotch_03.img',
            'Path hdr': path_folder160720 + r'\blotch_03.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:700, 60:130',
            'Class': 'blotch necrosis',
            'AOI': [[560, 566, 90, 94]]},
        {
            'Path image': path_folder160720 + r'\blotch_04.img',
            'Path hdr': path_folder160720 + r'\blotch_04.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:550, 70:130',
            'Class': 'blotch necrosis',
            'AOI': [[533, 536, 87, 89], [146, 152, 105, 107]]},
        {
            'Path image': path_folder160720 + r'\blotch_06.img',
            'Path hdr': path_folder160720 + r'\blotch_06.hdr',
            'Area w.r.': '70:380, 220:350',
            'Area leaf': '120:560, 60:130',
            'Class': 'blotch necrosis',
            'AOI': [[159, 168, 86, 88], [308, 310, 92, 93]]},
        {
            'Path image': path_folder160720 + r'\blotch_07.img',
            'Path hdr': path_folder160720 + r'\blotch_07.hdr',
            'Area w.r.': '70:380, 220:350',
            'Area leaf': '120:540, 60:120',
            'Class': 'blotch necrosis',
            'AOI': [[219, 225, 90, 91]]},

        # blotch chlorosis
        {
            'Path image': path_folder160720 + r'\blotch_01.img',
            'Path hdr': path_folder160720 + r'\blotch_01.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:550, 50:140',
            'Class': 'blotch chlorosis',
            'AOI': [[472, 473, 87, 89]]},
        {
            'Path image': path_folder160720 + r'\blotch_02.img',
            'Path hdr': path_folder160720 + r'\blotch_02.hdr',
            'Area w.r.': '70:380, 220:350',
            'Area leaf': '120:670, 50:140',
            'Class': 'blotch chlorosis',
            'AOI': [[238, 251, 100, 105]]},
        {
            'Path image': path_folder160720 + r'\blotch_04.img',
            'Path hdr': path_folder160720 + r'\blotch_04.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:550, 70:130',
            'Class': 'blotch chlorosis',
            'AOI': [[520, 525, 88, 90], [341, 345, 110, 112], [180, 200, 105,
                                                               107]]},
        {
            'Path image': path_folder160720 + r'\blotch_07.img',
            'Path hdr': path_folder160720 + r'\blotch_07.hdr',
            'Area w.r.': '70:380, 220:350',
            'Area leaf': '120:540, 60:120',
            'Class': 'blotch chlorosis',
            'AOI': [[205, 209, 87, 92], [390, 410, 87, 90]]},

        # healthy:
        {
            'Path image': path_folder160720 + r'\healthy_01.img',
            'Path hdr': path_folder160720 + r'\healthy_01.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:500, 50:150',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_02.img',
            'Path hdr': path_folder160720 + r'\healthy_02.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:780, 50:150',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_03.img',
            'Path hdr': path_folder160720 + r'\healthy_03.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '100:650, 50:150',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_04.img',
            'Path hdr': path_folder160720 + r'\healthy_04.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '100:680, 50:150',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_05.img',
            'Path hdr': path_folder160720 + r'\healthy_05.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '100:700, 50:150',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_06.img',
            'Path hdr': path_folder160720 + r'\healthy_06.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '90:450, 50:130',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_07.img',
            'Path hdr': path_folder160720 + r'\healthy_07.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '120:500, 50:130',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_08.img',
            'Path hdr': path_folder160720 + r'\healthy_08.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '110:640, 60:130',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_09.img',
            'Path hdr': path_folder160720 + r'\healthy_09.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '150:430, 60:130',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_10.img',
            'Path hdr': path_folder160720 + r'\healthy_10.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '130:520, 50:130',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_11.img',
            'Path hdr': path_folder160720 + r'\healthy_11.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '100:420, 50:130',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_12.img',
            'Path hdr': path_folder160720 + r'\healthy_12.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '80:650, 50:130',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder160720 + r'\healthy_13.img',
            'Path hdr': path_folder160720 + r'\healthy_13.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '100:430, 50:130',
            'Class': 'healthy',
            'AOI': []},
    ]

    other_leaves_160720_rust = [
        {
            'Path image': path_folder160720 + r'\rust_01.img',
            'Path hdr': path_folder160720 + r'\rust_01.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '100:550, 50:130',
            'Class': 'rust',
            'AOI': [],},
        {
            'Path image': path_folder160720 + r'\rust_04.img',
            'Path hdr': path_folder160720 + r'\rust_04.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '130:670, 50:130',
            'Class': 'rust',
            'AOI': [],},
        {
            'Path image': path_folder160720 + r'\rust_05.img',
            'Path hdr': path_folder160720 + r'\rust_05.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '130:480, 50:130',
            'Class': 'rust',
            'AOI': [],},
        {
            'Path image': path_folder160720 + r'\rust_09.img',
            'Path hdr': path_folder160720 + r'\rust_09.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '140:510, 50:140',
            'Class': 'rust',
            'AOI': []},
        {
            'Path image': path_folder230720 + r'\healthy_1.img',
            'Path hdr': path_folder230720 + r'\healthy_1.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '150:600, 110:170',
            'Class': 'rust',
            'AOI': []},
        {
            'Path image': path_folder230720 + r'\healthy_2.img',
            'Path hdr': path_folder230720 + r'\healthy_2.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '160:600, 80:150',
            'Class': 'healthy',
            'AOI': []},
        {
            'Path image': path_folder230720 + r'\rust_1.img',
            'Path hdr': path_folder230720 + r'\rust_1.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '150:810, 110:180',
            'Class': 'rust',
            'AOI': []},
        {
            'Path image': path_folder230720 + r'\rust_2.img',
            'Path hdr': path_folder230720 + r'\rust_2.hdr',
            'Area w.r.': '50:380, 220:350',
            'Area leaf': '150:730, 110:190',
            'Class': 'rust',
            'AOI': []},
        {
            'Path image': path_folder230720 + r'\blotch_1.img',
            'Path hdr': path_folder230720 + r'\blotch_1.hdr',
            'Area w.r.': '90:400, 240:360',
            'Area leaf': '170:680, 124:175',
            'Class': 'blotch',
            'AOI': []},
        {
            'Path image': path_folder230720 + r'\blotch_2.img',
            'Path hdr': path_folder230720 + r'\blotch_2.hdr',
            'Area w.r.': '90:400, 240:360',
            'Area leaf': '150:600, 110:180',
            'Class': 'blotch',
            'AOI': []},
    ]

    df = read_data_to_df(leaves_160720, snv_corr=False)
    df.to_pickle('leaf_dataframe_train16.csv', protocol=4)
    print('Done!')
