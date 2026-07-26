import os
import pickle
import cv2
import re
import glob
import numpy as np
import json
from i308_utils import show_images
from matplotlib import pyplot as plt
from src.utils import get_images_paths

DATASETS_PATH = "datasets"

def save_rectified_images(dataset_name):
    """
    Rectifica y guarda pares de imágenes estéreo, y exporta calibración rectificada.

    Lee calibración y mapas de remapeo desde pkl, aplica cv2.remap a cada par
    de capturas, guarda imágenes rectificadas en data/rect y genera
    stereo_calibration.json con parámetros intrínsecos y baseline.

    Args:
        dataset_name: Nombre del dataset dentro de datasets/.

    Returns:
        None.
    """
    calibration_path = f"{DATASETS_PATH}/{dataset_name}/data/stereo_calibration.pkl"
    with open(calibration_path, "rb") as f:
        calib = pickle.load(f)

    maps_path = f"{DATASETS_PATH}/{dataset_name}/data/stereo_maps.pkl"
    with open(maps_path, "rb") as f:
        maps = pickle.load(f)

    left_images_paths, right_images_paths = get_images_paths(f"{DATASETS_PATH}/{dataset_name}/data/captures")  

    print(left_images_paths, right_images_paths)

    for i in range(len(left_images_paths)):
        left_image = cv2.imread(left_images_paths[i], cv2.IMREAD_GRAYSCALE)
        right_image = cv2.imread(right_images_paths[i], cv2.IMREAD_GRAYSCALE)

        left_image_rectified = cv2.remap(left_image, maps['left_map_x'], maps['left_map_y'], cv2.INTER_LINEAR)
        right_image_rectified = cv2.remap(right_image, maps['right_map_x'], maps['right_map_y'], cv2.INTER_LINEAR)

        cv2.imwrite(f"{DATASETS_PATH}/{dataset_name}/data/rect/left_{i}.jpg", left_image_rectified)
        cv2.imwrite(f"{DATASETS_PATH}/{dataset_name}/data/rect/right_{i}.jpg", right_image_rectified)

    w, h = left_image.shape[1], left_image.shape[0]
    fx = calib['left_K'][0][0]
    fy = calib['left_K'][1][1]
    cx0 = calib['left_K'][0][2]
    cy0 = calib['left_K'][1][2]
    baseline = np.linalg.norm(calib['T'])

    calib_json = {
        "width": w,
        "height": h,
        "baseline_meters": baseline / 1000,
        "fx": fx,
        "fy": fy,
        "cx0": cx0,
        "cx1": cx0,
        "cy": cy0,
        "depth_range": [0.05, 20.0],
        "left_image_rect_normalized": [0, 0, 1, 1]
    }

    with open(f"{DATASETS_PATH}/{dataset_name}/data/rect/stereo_calibration.json", "w") as f:
        json.dump(calib_json, f)


def show_rectified_pair(dataset_name, image_index):
    """
    Muestra un par original y su versión rectificada con líneas guía horizontales.

    Args:
        dataset_name: Nombre del dataset dentro de datasets/.
        image_index: Índice del par a visualizar.

    Returns:
        None.
    """
    left_images_paths, right_images_paths = get_images_paths(f"{DATASETS_PATH}/{dataset_name}/data/captures")
    left_image = cv2.imread(left_images_paths[image_index], cv2.IMREAD_GRAYSCALE)
    right_image = cv2.imread(right_images_paths[image_index], cv2.IMREAD_GRAYSCALE)

    left_image_rect = cv2.imread(f"{DATASETS_PATH}/{dataset_name}/data/rect/left_{image_index}.jpg", cv2.IMREAD_GRAYSCALE)
    right_image_rect = cv2.imread(f"{DATASETS_PATH}/{dataset_name}/data/rect/right_{image_index}.jpg", cv2.IMREAD_GRAYSCALE)

    show_images([left_image, right_image], ["left", "right"])
    
    fig, axes = show_images([left_image_rect, right_image_rect], ["left_rectified", "right_rectified"], show=False)

    height = left_image.shape[0]
    lines = [int(height * 0.20), int(height * 0.5), int(height * 0.80)]

    for y, c in zip(lines, ['r', 'b', 'c']):
        axes[0].axhline(y=y, color=c, linestyle='-', linewidth=0.5)
        axes[1].axhline(y=y, color=c, linestyle='-', linewidth=0.5)
    plt.show()
