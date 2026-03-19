# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:10:12 2025

@author: pky0507
"""

import argparse
from tqdm import tqdm

import numpy as np
import cv2
import pydicom
from skimage.exposure import match_histograms
from multiprocessing import Pool, cpu_count

def list_all_files(root_dir):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.dcm'):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    return file_paths
    
def process(im, ref, img_path):
    if im.BitsStored == 14 or 'Malign' in img_path and any(i in img_path for i in ['96', '98']):
        im = match_foreground_histogram(im, ref)
        im.BitsAllocated = ref.BitsAllocated
        im.BitsStored = ref.BitsStored
        im.HighBit = ref.HighBit
    return im

def match_foreground_histogram(im, ref):
    """
    Match the histogram of the foreground in `source` to the foreground in `reference`.

    Parameters:
        source (np.ndarray): Source image to be adjusted
        reference (np.ndarray): Reference image whose histogram we want to match

    Returns:
        np.ndarray: Histogram-matched image (foreground only modified)
    """
    source = im.pixel_array.astype(np.float32)
    reference = ref.pixel_array.astype(np.float32)

    # Identify foreground pixels
    mask_source = source > 0
    mask_ref = reference > 0

    # Match only foreground
    matched_foreground = match_histograms(
        source[mask_source], reference[mask_ref]
    )

    # Replace foreground in the original image
    matched = np.copy(source)
    matched[mask_source] = matched_foreground
    matched = matched.astype(im.pixel_array.dtype)
    im.PixelData = matched.tobytes()
    return im

def process_file(args_tuple):
    img_path, args, ref = args_tuple
    output_path = os.path.dirname(img_path).replace(args.data_path, args.output_dir)
    os.makedirs(output_path, exist_ok=True)
    output_path = os.path.join(output_path, os.path.basename(img_path))
    if 'Benign' in img_path:
        output_path_png = os.path.join(args.output_dir+'_PNG', "Benign", os.path.basename(os.path.dirname(img_path)))+os.path.basename(img_path).replace('.dcm', '.png')
    else:
        output_path_png = os.path.join(args.output_dir+'_PNG', "Malign", os.path.basename(os.path.dirname(img_path)))+os.path.basename(img_path).replace('.dcm', '.png') 
    im = pydicom.dcmread(img_path)
    im = process(im, ref, img_path)
        
    im.save_as(output_path) # Save the normalized and resized DICOM file
    cv2.imwrite(output_path_png, (im.pixel_array / (2**im.BitsStored-1) * 65535).astype(np.uint16))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MammoFL")
    parser.add_argument("--data-path", default="/dataset/Mammogram/LUMINA_RAW", type=str, help="dataset path")
    parser.add_argument("-o", "--output-dir", default="./LUMINA", type=str, help="path to save outputs")
    
    args = parser.parse_args()
    count = 0
    image = list_all_files(args.data_path)
    ref = pydicom.dcmread(os.path.join(args.data_path,'Malign/8/L_CC.dcm'))
    os.makedirs(os.path.join(args.output_dir+'_PNG', "Benign"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir+'_PNG', "Malign"), exist_ok=True)
    # Run multiprocessing
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, [(img_path, args, ref) for img_path in image]), total=len(image)))
    #     cv2.imwrite(output_path_png, (im.pixel_array / (2**13-1) * 255).astype(np.uint8))
