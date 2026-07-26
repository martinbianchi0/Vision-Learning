import json
import pickle
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from stereodemo.method_cre_stereo import CREStereo
from stereodemo.methods import Calibration, Config, InputPair
from .aruco import (create_charuco_board, detect_charuco_markers,
                    estimate_camera_pose_with_homography)
from .utils import get_images_paths

DATASETS_PATH = "datasets"

def reconstruction_from_dataset(dataset_name):
    """
    Reconstruye una nube de puntos 3D y estima poses de cámara a partir de un dataset.

    Carga calibración rectificada y matriz Q, calcula disparidad con CREStereo para
    cada par de imágenes rectificadas, reproyecta a 3D, estima la pose de cámara
    respecto a un tablero Charuco y transforma puntos al sistema del mundo. Guarda
    puntos acumulados y poses en archivos .npy dentro de data/results.

    Args:
        dataset_name: Nombre del dataset dentro de datasets/.

    Returns:
        None. Escribe 'points.npy' y 'poses.npy' en el directorio de resultados.
    """
    json_path = f"{DATASETS_PATH}/{dataset_name}/data/rect/stereo_calibration.json"
    maps_path = f"{DATASETS_PATH}/{dataset_name}/data/stereo_maps.pkl"

    with open(json_path, "r") as f:
        data = json.load(f)

    with open(maps_path, "rb") as f:
        maps = pickle.load(f)

    calibration = Calibration(**data)
    Q = maps['Q']

    rectified_K = maps['P1'][:, :3]
    dist_coeffs = None
    left_calibration_rectified = (rectified_K, dist_coeffs)

    board = create_charuco_board(
        squares_x=5,
        squares_y=7,
        square_length=52.6,
        marker_length=31.3,
    )

    models_path = Path('models')
    config = Config(models_path=models_path)
    method = CREStereo(config)

    left_images_paths, right_images_paths = get_images_paths(
        f"{DATASETS_PATH}/{dataset_name}/data/rect")

    accumulated_points_world = []
    camera_poses = []

    print(f"Iniciando procesamiento de {len(left_images_paths)} pares de imágenes...")
    print("--- Presiona 'ESC' en las ventanas de imagen para saltar al final ---")

    for i in range(len(left_images_paths)):
        print(f"\n--- Procesando Frame {i+1}/{len(left_images_paths)} ---")
        left_image_rect = cv2.imread(
            left_images_paths[i], cv2.IMREAD_GRAYSCALE)
        right_image_rect = cv2.imread(
            right_images_paths[i], cv2.IMREAD_GRAYSCALE)

        pair = InputPair(left_image_rect,
                         right_image_rect, calibration, "status?")
        disparity = method.compute_disparity(pair)
        disp = disparity.disparity_pixels

        points3d_camera = cv2.reprojectImageTo3D(disp, Q)
        depth_m = points3d_camera[:, :, 2]
        mask = disp > 0
        points3d_camera_valid = points3d_camera[mask]

        near, far = np.nanpercentile(depth_m, [2, 98])
        plt.figure(figsize=(10,5))
        im = plt.imshow(depth_m, vmin=near, vmax=far)
        plt.colorbar(im, label='Profundidad [mm]')
        plt.axis('off'); plt.show()

        detection = detect_charuco_markers(left_image_rect, board)
        
        pose_result = estimate_camera_pose_with_homography(
            left_image_rect,
            board,
            detection,
            left_calibration_rectified,
        )

        if pose_result is None:
            print(f"(!) ADVERTENCIA: No se detectó el tablero en el frame {i}. Saltando.")
            continue
        
        ok, rvec, tvec = pose_result

        c_R_o, _ = cv2.Rodrigues(rvec)
        c_T_o = np.vstack((np.column_stack((c_R_o, tvec)), [0, 0, 0, 1]))
        o_T_c = np.linalg.inv(c_T_o)

        camera_position = o_T_c[0:3, 3]
        print(f"(i) POSE OK. Posición de cámara (X,Y,Z en milimetros): {np.round(camera_position, 2)}")

        camera_poses.append(o_T_c)

        points3d_camera_h = np.vstack(
            (points3d_camera_valid.T, np.ones(points3d_camera_valid.shape[0])))
        points3d_world = (o_T_c @ points3d_camera_h)[:3, :].T

        print(f"(i) PUNTOS 3D generados en este frame: {len(points3d_world)}")

        accumulated_points_world.append(points3d_world)


    if not accumulated_points_world:
        print("No se acumularon puntos.")
        return

    print("Combinando nubes de puntos...")
    final_point_cloud = np.vstack(accumulated_points_world)

    final_poses_array = np.stack(camera_poses)


    points_filename = f"{DATASETS_PATH}/{dataset_name}/data/results/points.npy"
    poses_filename = f"{DATASETS_PATH}/{dataset_name}/data/results/poses.npy"

    np.save(points_filename, final_point_cloud)
    np.save(poses_filename, final_poses_array)

    print(f"Puntos guardados en: {points_filename}")
    print(f"Poses guardadas en: {poses_filename}")
    