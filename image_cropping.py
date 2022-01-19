
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