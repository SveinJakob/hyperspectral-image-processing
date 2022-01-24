# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 2022 19:00

@author: Svein Jakob Kristoffersen (sveinjakobkristoffersen@gmail.com)

Functions used to apply corrections to single spectrum or list of spectrum.

"""

import numpy as np
import numpy.linalg as la
from math import log10
from scipy.signal import savgol_filter





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


def emsc(X, d=0, ref_spec=None):
    """
    Basic EMSC algorithm.

    Parameters
    ----------
    X : Matrix of spectra to be corrected
    d : Maximum Order of the polynomial trends to correct for.
        Default is 0 resulting in basic MSC correction with no polynomial
        trend correction.
    ref_spec : Array with the spectrum to use as the reference.
        If not specified, the mean spectrum is used.

    Returns
    -------
    Xprep : Matrix with the corrected spectra.
    """

    if ref_spec is None:
        Xref = np.mean(X, axis=0)  # Reference spectrum
    else:
        Xref = ref_spec

    n = len(Xref)  # Number of features
    P = np.zeros([n, d + 2])  # Polynomial basis vectors
    P[:, 0] = Xref  # Place the reference spectrum at the first columns
    P[:, 1] = np.ones_like(Xref)  # Place the baseline at the second column
    for i in range(d):
        P[:, i + 2] = np.linspace(-1, 1, n) ** (i + 1)  # Add the other polynomial basis vectors from column three
    coeffs, _, _, _ = la.lstsq(P, X.T, rcond=None)  # Solve the least squares problem to determine the coefficients
    Xprep = X.copy()
    Xprep -= (P[:, 1:] @ coeffs[1:, :]).T  # Subtract the unwanted multiplicative effects
    Xprep /= coeffs[0, :].reshape(-1, 1)  # Divide by the coefficients for the reference spectrum

    return Xprep


def snv_single_spectrum(spectrum):
    """Perform standard normal variate correction on a single spectrum"""
    return (spectrum - np.mean(spectrum)) / np.std(spectrum)


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