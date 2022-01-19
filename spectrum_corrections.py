



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