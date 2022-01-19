# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 2022 19:00

@author: Svein Jakob Kristoffersen (sveinjakobkristoffersen@gmail.com)

Functions used to apply corrections to images.

"""


def absorbance_matrix(datamatrix):
    """
    Functino to convert matrix of spectra (reflectance) to matrix of
    absorbance spectra.

    """
    new_matrix = np.zeros_like(datamatrix)
    for i, spectrum in enumerate(datamatrix):
        new_matrix[i, :] = absorbance_spectrum(spectrum)
    return new_matrix


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


def snv_on_image(img):
    """
    Performs standard normal variate correction correction on an entire image

    """
    img_corr = np.zeros(shape=(img.shape[0], img.shape[1], img.shape[2]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img_corr[i, j, :] = snv_single_spectrum(img[i, j, :])
    return np.array(img_corr)