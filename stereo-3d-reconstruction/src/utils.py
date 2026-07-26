import cv2
import numpy as np
import glob
import os
import re

def get_images_paths(calib_dir):
    """
    Devuelve listas pareadas de rutas de imágenes izquierda/derecha en un directorio.

    Busca archivos con patrón left_*.{jpg,png} y right_*.{jpg,png}, emparejando por índice
    numérico coherente en el nombre, y devuelve listas ordenadas y alineadas.

    Args:
        calib_dir: Ruta del directorio con capturas estéreo.

    Returns:
        Tuple (left_paths, right_paths) con rutas alineadas por índice.
    """
    left_paths  = glob.glob(os.path.join(calib_dir, "left_*.*"))
    right_paths = glob.glob(os.path.join(calib_dir, "right_*.*"))

    pat = re.compile(r"(left|right)_(\d+)\.(jpg|png)$", re.IGNORECASE)
    def build_index(files):
        out = {}
        for p in files:
            m = pat.search(os.path.basename(p))
            if m:
                out[int(m.group(2))] = p
        return out

    L = build_index(left_paths)
    R = build_index(right_paths)
    common_idx = sorted(set(L) & set(R))

    left_images_paths = [L[i] for i in common_idx]
    right_images_paths = [R[i] for i in common_idx]

    return left_images_paths, right_images_paths

def draw_line(img, pt1, pt2, color, thickness=3):
    """
    Dibuja una línea entre dos puntos en una imagen.

    Args:
        img: Imagen destino (modificada in-place).
        pt1: Punto inicial (x, y) en float o int.
        pt2: Punto final (x, y) en float o int.
        color: Color BGR de la línea.
        thickness: Grosor de la línea.

    Returns:
        Imagen con la línea dibujada.
    """
    pt1 = (np.round(pt1[0]).astype(int), np.round(pt1[1]).astype(int))
    pt2 = (np.round(pt2[0]).astype(int), np.round(pt2[1]).astype(int))
    ret = cv2.line(img, pt1, pt2, color, thickness)
    return ret


def plot_axis(
        image,
        calibration,
        pose,
        axis_len=100,
        right_handed=False,
        thickness=3,
        distorts=True
):
    """
    Proyecta y dibuja los ejes XYZ de un marco en una imagen.

    Args:
        image: Imagen BGR donde dibujar.
        calibration: Tupla (K, dist_coeffs).
        pose: Tupla (ok, rvec, tvec) de la cámara respecto al marco.
        axis_len: Longitud de los ejes en unidades del mundo.
        right_handed: Si True, invierte el eje Z para sistema derecho.
        thickness: Grosor de las líneas.
        distorts: Si True, usa distorsión al proyectar.

    Returns:
        Imagen con ejes dibujados o None si la pose no es válida.
    """
    pose_ok, rv, tv = pose
    if not pose_ok:
        return

    K, dist_coeffs = calibration

    z_dir = -1 if right_handed else 1

    axis_points = np.array([
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, z_dir * axis_len]
    ], dtype=np.float32)


    if distorts:
        use_dist_coeffs = dist_coeffs
    else:
        use_dist_coeffs = None
    axis_points_proj, _ = cv2.projectPoints(axis_points, rv, tv, K, use_dist_coeffs)

    axis_points_proj = axis_points_proj.reshape(-1, 2)
    origin = tuple(axis_points_proj[0].ravel())

    draw_line(image, origin, axis_points_proj[1], (0, 0, 255), thickness=thickness)
    draw_line(image, origin, axis_points_proj[2], (0, 255, 0), thickness=thickness)
    draw_line(image, origin, axis_points_proj[3], (255, 0, 0), thickness=thickness)

    return image