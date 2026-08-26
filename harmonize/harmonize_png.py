# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 15:10:12 2025

@author: pky0507
"""

import os
import argparse
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
from skimage.exposure import match_histograms
from multiprocessing import Pool, cpu_count

def list_all_files(root_dir):
    file_paths = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.png'):
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    return file_paths

def match_foreground_histogram(src_arr, ref_arr, ratio=1.0):
    """Foreground-isolated histogram matching supporting arbitrary data types,

    image shapes, and linear blending control.
    """
    src_mask = src_arr > 0
    ref_mask = ref_arr > 0

    if not np.any(src_mask) or not np.any(ref_mask):
        return src_arr.copy()

    src_pixels = src_arr[src_mask]
    ref_pixels = ref_arr[ref_mask]

    matched_pixels = match_histograms(src_pixels, ref_pixels, channel_axis=None)

    # Blend using float operations to prevent premature overflow/underflow
    blended_pixels = (1.0 - ratio) * src_pixels.astype(np.float64) + ratio * matched_pixels.astype(np.float64)

    # Determine dynamic clipping bounds based on src_arr dtype
    src_dtype = src_arr.dtype
    if np.issubdtype(src_dtype, np.integer):
        info = np.iinfo(src_dtype)
        clipped_pixels = np.clip(np.round(blended_pixels), info.min, info.max).astype(src_dtype)
    elif np.issubdtype(src_dtype, np.floating):
        clipped_pixels = blended_pixels.astype(src_dtype)
    else:
        clipped_pixels = blended_pixels.astype(src_dtype)

    output_img = src_arr.copy()
    output_img[src_mask] = clipped_pixels

    return output_img

def process_file(args_tuple):
    img_path, args, ref = args_tuple
    output_path = os.path.dirname(img_path).replace(args.data_path, args.output_dir)
    output_path = os.path.join(output_path, os.path.basename(img_path))
    if 'Benign' in img_path:
        output_path_png = os.path.join(args.output_dir+'_PNG', "Benign", os.path.basename(os.path.dirname(img_path)))+os.path.basename(img_path).replace('.dcm', '.png')
    else:
        output_path_png = os.path.join(args.output_dir+'_PNG', "Malign", os.path.basename(os.path.dirname(img_path)))+os.path.basename(img_path).replace('.dcm', '.png') 
    im = np.array(Image.open(img_path))
    output_img = match_foreground_histogram(im, ref)
    cv2.imwrite(output_path_png, output_img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MammoFL")
    parser.add_argument("--data-path", default="/dataset/Mammogram/LUMINA_RAW_PNG", type=str, help="dataset path")
    parser.add_argument("-o", "--output-dir", default="./LUMINA", type=str, help="path to save outputs")
    parser.add_argument("-r", "--ref", default="/dataset/Mammogram/LUMINA_RAW_PNG/Malign/8L_CC.png", type=str, help="reference image path")
    
    args = parser.parse_args()
    image = list_all_files(args.data_path)
    ref = np.array(Image.open(args.ref))
    os.makedirs(os.path.join(args.output_dir+'_PNG', "Benign"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir+'_PNG', "Malign"), exist_ok=True)
    # Run multiprocessing
    with Pool(processes=cpu_count()) as pool:
        list(tqdm(pool.imap_unordered(process_file, [(img_path, args, ref) for img_path in image]), total=len(image)))