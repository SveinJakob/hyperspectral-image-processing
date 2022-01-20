# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 2022 19:00

@author: Svein Jakob Kristoffersen (sveinjakobkristoffersen@gmail.com)

Example use of functions

"""

from processsing_functions import preprocess_image


image_info = {'Path image': r'C:\Users\svein\Downloads\drive-download-20220120T185236Z-001\rust_12.img',
              'Path hdr': r'C:\Users\svein\Downloads\drive-download-20220120T185236Z-001\rust_12.hdr',
              'Area w.r.': '50:380, 220:350',
              'Area leaf': '100:480, 50:140',
              }
emsc_info = {'ref_spectrum': None,
             'degree': None,
             }

img, mask = preprocess_image(image_info)

