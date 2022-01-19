
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
