import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as Rot


def load_calibration_data(dataset_name):
    """Carga los parámetros de calibración y mapas de un dataset"""
    base_path = Path(f"datasets/{dataset_name}/data")

    with open(base_path / "stereo_calibration.pkl", "rb") as f:
        calib = pickle.load(f)

    with open(base_path / "stereo_maps.pkl", "rb") as f:
        maps = pickle.load(f)

    return calib, maps


def analyze_intrinsics(calib, verbose=True):
    """Analiza los parámetros intrínsecos de ambas cámaras

    Args:
        calib: Diccionario con parámetros de calibración
        verbose: Si True, imprime información detallada

    Returns:
        dict: Diccionario con métricas de los parámetros intrínsecos
    """
    K1 = calib["left_K"]
    K2 = calib["right_K"]

    metrics = {
        "K1": K1,
        "K2": K2,
        "fx1": K1[0, 0],
        "fy1": K1[1, 1],
        "cx1": K1[0, 2],
        "cy1": K1[1, 2],
        "fx2": K2[0, 0],
        "fy2": K2[1, 1],
        "cx2": K2[0, 2],
        "cy2": K2[1, 2],
        "fx_ratio1": K1[0, 0] / K1[1, 1],
        "fx_ratio2": K2[0, 0] / K2[1, 1],
        "fx_diff_pct": abs(K1[0, 0] - K2[0, 0]) / K1[0, 0] * 100,
        "fy_diff_pct": abs(K1[1, 1] - K2[1, 1]) / K1[1, 1] * 100,
    }

    if verbose:
        print("=" * 60)
        print("PARÁMETROS INTRÍNSECOS - CÁMARA IZQUIERDA")
        print("=" * 60)
        print(f"\nDistancia focal fx: {metrics['fx1']:.2f} px")
        print(f"Distancia focal fy: {metrics['fy1']:.2f} px")
        print(f"Relación fx/fy: {metrics['fx_ratio1']:.4f}")
        print(f"Centro óptico (cx, cy): ({metrics['cx1']:.2f}, {metrics['cy1']:.2f})")

        print("\n" + "=" * 60)
        print("PARÁMETROS INTRÍNSECOS - CÁMARA DERECHA")
        print("=" * 60)
        print(f"\nDistancia focal fx: {metrics['fx2']:.2f} px")
        print(f"Distancia focal fy: {metrics['fy2']:.2f} px")
        print(f"Relación fx/fy: {metrics['fx_ratio2']:.4f}")
        print(f"Centro óptico (cx, cy): ({metrics['cx2']:.2f}, {metrics['cy2']:.2f})")

        print("\n" + "=" * 60)
        print("COMPARACIÓN ENTRE CÁMARAS")
        print("=" * 60)
        print(
            f"Diferencia en fx: {abs(K1[0, 0] - K2[0, 0]):.2f} px ({metrics['fx_diff_pct']:.2f}%)"
        )
        print(
            f"Diferencia en fy: {abs(K1[1, 1] - K2[1, 1]):.2f} px ({metrics['fy_diff_pct']:.2f}%)"
        )

    return metrics


def calculate_fov(K, img_width=1280, img_height=720):
    """Calcula el campo de visión (FOV) de una cámara

    Args:
        K: Matriz de cámara
        img_width: Ancho de la imagen en píxeles
        img_height: Alto de la imagen en píxeles

    Returns:
        tuple: (fov_horizontal, fov_vertical) en grados
    """
    fov_h = 2 * np.arctan(img_width / (2 * K[0, 0])) * 180 / np.pi
    fov_v = 2 * np.arctan(img_height / (2 * K[1, 1])) * 180 / np.pi
    return fov_h, fov_v


def analyze_distortion(calib, verbose=True, plot=False):
    """Analiza los coeficientes de distorsión

    Args:
        calib: Diccionario con parámetros de calibración
        verbose: Si True, imprime información detallada
        plot: Si True, genera gráficos de distorsión

    Returns:
        dict: Diccionario con coeficientes de distorsión
    """
    D1 = calib["left_dist"]
    D2 = calib["right_dist"]

    metrics = {
        "D1": D1,
        "D2": D2,
        "k1_left": D1[0, 0],
        "k2_left": D1[1, 0],
        "p1_left": D1[2, 0],
        "p2_left": D1[3, 0],
        "k1_right": D2[0, 0],
        "k2_right": D2[1, 0],
        "p1_right": D2[2, 0],
        "p2_right": D2[3, 0],
    }

    if len(D1) > 4:
        metrics["k3_left"] = D1[4, 0]
    if len(D2) > 4:
        metrics["k3_right"] = D2[4, 0]

    if verbose:
        print("=" * 60)
        print("COEFICIENTES DE DISTORSIÓN")
        print("=" * 60)

        print("\nCámara Izquierda (D1):")
        print(f"  k1 (radial): {metrics['k1_left']:.6f}")
        print(f"  k2 (radial): {metrics['k2_left']:.6f}")
        print(f"  p1 (tangencial): {metrics['p1_left']:.6f}")
        print(f"  p2 (tangencial): {metrics['p2_left']:.6f}")
        if "k3_left" in metrics:
            print(f"  k3 (radial): {metrics['k3_left']:.6f}")

        print("\nCámara Derecha (D2):")
        print(f"  k1 (radial): {metrics['k1_right']:.6f}")
        print(f"  k2 (radial): {metrics['k2_right']:.6f}")
        print(f"  p1 (tangencial): {metrics['p1_right']:.6f}")
        print(f"  p2 (tangencial): {metrics['p2_right']:.6f}")
        if "k3_right" in metrics:
            print(f"  k3 (radial): {metrics['k3_right']:.6f}")

    if plot:
        plot_distortion_coefficients(D1, D2)

    return metrics


def plot_distortion_coefficients(D1, D2):
    """Visualiza los coeficientes de distorsión"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    coeffs_names = ["k1", "k2", "p1", "p2", "k3"][: len(D1)]
    d1_values = D1.flatten()[:5]
    d2_values = D2.flatten()[:5]

    x = np.arange(len(coeffs_names))
    width = 0.35

    axes[0].bar(x - width / 2, d1_values, width, label="Cámara Izquierda", alpha=0.8)
    axes[0].bar(x + width / 2, d2_values, width, label="Cámara Derecha", alpha=0.8)
    axes[0].set_xlabel("Coeficiente")
    axes[0].set_ylabel("Valor")
    axes[0].set_title("Coeficientes de Distorsión")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(coeffs_names)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].axhline(y=0, color="k", linestyle="-", linewidth=0.5)

    axes[1].bar(
        x - width / 2, np.abs(d1_values), width, label="Cámara Izquierda", alpha=0.8
    )
    axes[1].bar(
        x + width / 2, np.abs(d2_values), width, label="Cámara Derecha", alpha=0.8
    )
    axes[1].set_xlabel("Coeficiente")
    axes[1].set_ylabel("Magnitud (escala log)")
    axes[1].set_title("Magnitud de Distorsión (valor absoluto)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(coeffs_names)
    axes[1].set_yscale("log")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_extrinsics(calib, verbose=True):
    """Analiza los parámetros extrínsecos (geometría estéreo)

    Args:
        calib: Diccionario con parámetros de calibración
        verbose: Si True, imprime información detallada

    Returns:
        dict: Diccionario con métricas de parámetros extrínsecos
    """
    R = calib["R"]
    T = calib["T"]

    baseline_mm = np.linalg.norm(T)
    baseline_m = baseline_mm / 1000.0

    # Convertir rotación a ángulos de Euler
    r = Rot.from_matrix(R)
    euler_angles = r.as_euler("xyz", degrees=True)

    metrics = {
        "R": R,
        "T": T,
        "baseline_mm": baseline_mm,
        "baseline_m": baseline_m,
        "det_R": np.linalg.det(R),
        "euler_roll": euler_angles[0],
        "euler_pitch": euler_angles[1],
        "euler_yaw": euler_angles[2],
        "max_euler_angle": np.max(np.abs(euler_angles)),
    }

    if verbose:
        print("=" * 60)
        print("PARÁMETROS EXTRÍNSECOS")
        print("=" * 60)

        print(f"\nDeterminante de R: {metrics['det_R']:.6f} (debe ser ≈ 1.0)")

        print(f"\nÁngulos de Euler (xyz):")
        print(f"  Roll (x):  {metrics['euler_roll']:.2f}°")
        print(f"  Pitch (y): {metrics['euler_pitch']:.2f}°")
        print(f"  Yaw (z):   {metrics['euler_yaw']:.2f}°")

        print(f"\nBaseline (distancia entre cámaras):")
        print(f"  {baseline_mm:.2f} mm = {baseline_m:.4f} m")

        print("\n" + "=" * 60)
        print("VERIFICACIÓN")
        print("=" * 60)

        if np.allclose(R, np.eye(3), atol=0.1):
            print("✓ La matriz R está cerca de la identidad (buena alineación)")
        else:
            print("⚠ La matriz R se desvía de la identidad (posible desalineación)")

        max_angle = metrics["max_euler_angle"]
        if max_angle < 5:
            print(f"✓ Ángulos de rotación pequeños (máx: {max_angle:.2f}°)")
        elif max_angle < 15:
            print(f"⚠ Ángulos de rotación moderados (máx: {max_angle:.2f}°)")
        else:
            print(f"⚠ Ángulos de rotación grandes (máx: {max_angle:.2f}°)")

    return metrics


def analyze_rectified_projection(calib, maps, verbose=True):
    """Analiza las matrices de proyección rectificadas

    Args:
        calib: Diccionario con parámetros de calibración
        maps: Diccionario con mapas de rectificación
        verbose: Si True, imprime información detallada

    Returns:
        dict: Diccionario con métricas de proyección rectificada
    """
    P1 = maps.get("P1", calib.get("P1"))
    P2 = maps.get("P2", calib.get("P2"))
    T = calib["T"]

    f_rect = P1[0, 0]
    cx_rect = P1[0, 2]
    cy_rect = P1[1, 2]

    f2_rect = P2[0, 0]
    cx2_rect = P2[0, 2]
    cy2_rect = P2[1, 2]

    Tx = P2[0, 3]
    baseline_from_P2 = -Tx / f_rect
    baseline_from_T = np.linalg.norm(T) / 1000.0

    metrics = {
        "P1": P1,
        "P2": P2,
        "f_rect": f_rect,
        "cx_rect": cx_rect,
        "cy_rect": cy_rect,
        "f_diff": abs(f_rect - f2_rect),
        "cx_diff": abs(cx_rect - cx2_rect),
        "cy_diff": abs(cy_rect - cy2_rect),
        "baseline_from_P2": baseline_from_P2,
        "baseline_from_T": baseline_from_T,
        "baseline_diff": abs(baseline_from_P2 - baseline_from_T),
    }

    if verbose:
        print("=" * 60)
        print("MATRICES DE PROYECCIÓN RECTIFICADAS")
        print("=" * 60)

        print(f"\nDistancia focal rectificada (f): {f_rect:.2f} px")
        print(f"Centro óptico (cx, cy): ({cx_rect:.2f}, {cy_rect:.2f})")

        print(f"\nVerificación de alineación:")
        print(f"  Diferencia en f: {metrics['f_diff']:.6f} px")
        print(f"  Diferencia en cx: {metrics['cx_diff']:.6f} px")
        print(f"  Diferencia en cy: {metrics['cy_diff']:.6f} px")

        if metrics["cy_diff"] < 0.1:
            print("  ✓ Líneas epipolares correctamente alineadas horizontalmente")
        else:
            print("  ⚠ Posible desalineación en líneas epipolares")

        print(f"\nBaseline verificado desde P2: {baseline_from_P2:.4f} m")
        print(f"Baseline desde T: {baseline_from_T:.4f} m")
        print(f"Diferencia: {metrics['baseline_diff']:.6f} m")

        if metrics["baseline_diff"] / baseline_from_T < 0.01:
            print("✓ Baseline consistente entre P2 y T")
        else:
            print("⚠ Discrepancia en baseline")

    return metrics


def analyze_rectification_maps(maps, verbose=True, plot=False):
    """Analiza los mapas de rectificación

    Args:
        maps: Diccionario con mapas de rectificación
        verbose: Si True, imprime información detallada
        plot: Si True, genera visualizaciones

    Returns:
        dict: Diccionario con métricas de los mapas
    """
    left_map_x = maps["left_map_x"]
    left_map_y = maps["left_map_y"]
    right_map_x = maps["right_map_x"]
    right_map_y = maps["right_map_y"]

    h, w = left_map_x.shape
    y_grid, x_grid = np.mgrid[0:h, 0:w].astype(np.float32)

    # Calcular desplazamientos
    disp_left_x = left_map_x - x_grid
    disp_left_y = left_map_y - y_grid
    disp_left_mag = np.sqrt(disp_left_x**2 + disp_left_y**2)

    disp_right_x = right_map_x - x_grid
    disp_right_y = right_map_y - y_grid
    disp_right_mag = np.sqrt(disp_right_x**2 + disp_right_y**2)

    roi1 = maps.get("validRoi1", None)
    roi2 = maps.get("validRoi2", None)

    metrics = {
        "map_shape": left_map_x.shape,
        "disp_left_x_range": (disp_left_x.min(), disp_left_x.max()),
        "disp_left_y_range": (disp_left_y.min(), disp_left_y.max()),
        "disp_left_mag_mean": disp_left_mag.mean(),
        "disp_left_mag_max": disp_left_mag.max(),
        "disp_right_x_range": (disp_right_x.min(), disp_right_x.max()),
        "disp_right_y_range": (disp_right_y.min(), disp_right_y.max()),
        "disp_right_mag_mean": disp_right_mag.mean(),
        "disp_right_mag_max": disp_right_mag.max(),
        "roi1": roi1,
        "roi2": roi2,
    }

    if verbose:
        print("=" * 60)
        print("ANÁLISIS DE MAPAS DE RECTIFICACIÓN")
        print("=" * 60)

        print(f"\nDimensiones de los mapas: {metrics['map_shape']}")

        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DE DESPLAZAMIENTO - CÁMARA IZQUIERDA")
        print("=" * 60)
        print(
            f"Desplazamiento en X: [{disp_left_x.min():.2f}, {disp_left_x.max():.2f}] px"
        )
        print(
            f"Desplazamiento en Y: [{disp_left_y.min():.2f}, {disp_left_y.max():.2f}] px"
        )
        print(f"Magnitud promedio: {metrics['disp_left_mag_mean']:.2f} px")
        print(f"Magnitud máxima: {metrics['disp_left_mag_max']:.2f} px")

        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DE DESPLAZAMIENTO - CÁMARA DERECHA")
        print("=" * 60)
        print(
            f"Desplazamiento en X: [{disp_right_x.min():.2f}, {disp_right_x.max():.2f}] px"
        )
        print(
            f"Desplazamiento en Y: [{disp_right_y.min():.2f}, {disp_right_y.max():.2f}] px"
        )
        print(f"Magnitud promedio: {metrics['disp_right_mag_mean']:.2f} px")
        print(f"Magnitud máxima: {metrics['disp_right_mag_max']:.2f} px")

        if roi1 is not None and roi2 is not None:
            x1, y1, w1, h1 = roi1
            x2, y2, w2, h2 = roi2

            print("\n" + "=" * 60)
            print("REGIONES DE INTERÉS VÁLIDAS (ROI)")
            print("=" * 60)
            print(f"ROI Izquierda: x={x1}, y={y1}, w={w1}, h={h1}")
            print(
                f"  Área válida: {w1 * h1} px ({w1 * h1 / (w * h) * 100:.1f}% de la imagen)"
            )
            print(f"\nROI Derecha: x={x2}, y={y2}, w={w2}, h={h2}")
            print(
                f"  Área válida: {w2 * h2} px ({w2 * h2 / (w * h) * 100:.1f}% de la imagen)"
            )

            x = max(x1, x2)
            y = max(y1, y2)
            xe = min(x1 + w1, x2 + w2)
            ye = min(y1 + h1, y2 + h2)
            w_inter = xe - x
            h_inter = ye - y

            print(f"\nROI Intersección: w={w_inter}, h={h_inter}")
            print(
                f"  Área común: {w_inter * h_inter} px ({w_inter * h_inter / (w * h) * 100:.1f}% de la imagen)"
            )

    if plot:
        plot_rectification_maps(
            disp_left_x,
            disp_left_y,
            disp_left_mag,
            disp_right_x,
            disp_right_y,
            disp_right_mag,
        )

    return metrics


def plot_rectification_maps(
    disp_left_x, disp_left_y, disp_left_mag, disp_right_x, disp_right_y, disp_right_mag
):
    """Visualiza los mapas de rectificación"""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    im0 = axes[0, 0].imshow(disp_left_x, cmap="RdBu_r", vmin=-50, vmax=50)
    axes[0, 0].set_title("Desplazamiento X - Izquierda")
    axes[0, 0].axis("off")
    plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[1, 0].imshow(disp_right_x, cmap="RdBu_r", vmin=-50, vmax=50)
    axes[1, 0].set_title("Desplazamiento X - Derecha")
    axes[1, 0].axis("off")
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)

    im2 = axes[0, 1].imshow(disp_left_y, cmap="RdBu_r", vmin=-50, vmax=50)
    axes[0, 1].set_title("Desplazamiento Y - Izquierda")
    axes[0, 1].axis("off")
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im3 = axes[1, 1].imshow(disp_right_y, cmap="RdBu_r", vmin=-50, vmax=50)
    axes[1, 1].set_title("Desplazamiento Y - Derecha")
    axes[1, 1].axis("off")
    plt.colorbar(im3, ax=axes[1, 1], fraction=0.046, pad=0.04)

    im4 = axes[0, 2].imshow(disp_left_mag, cmap="hot")
    axes[0, 2].set_title("Magnitud - Izquierda")
    axes[0, 2].axis("off")
    plt.colorbar(im4, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im5 = axes[1, 2].imshow(disp_right_mag, cmap="hot")
    axes[1, 2].set_title("Magnitud - Derecha")
    axes[1, 2].axis("off")
    plt.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

    plt.suptitle(
        "Mapas de Rectificación: Desplazamientos desde Coordenadas Originales",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()


def calibration_summary(dataset_name, verbose=False):
    """Genera un resumen completo de la calibración de un dataset

    Args:
        dataset_name: Nombre del dataset
        verbose: Si True, imprime detalles

    Returns:
        dict: Diccionario con todas las métricas de calibración
    """
    calib, maps = load_calibration_data(dataset_name)

    summary = {
        "dataset": dataset_name,
        "intrinsics": analyze_intrinsics(calib, verbose=verbose),
        "distortion": analyze_distortion(calib, verbose=verbose, plot=False),
        "extrinsics": analyze_extrinsics(calib, verbose=verbose),
        "projection": analyze_rectified_projection(calib, maps, verbose=verbose),
        "maps": analyze_rectification_maps(maps, verbose=verbose, plot=False),
    }

    # Calcular FOV
    K1 = calib["left_K"]
    K2 = calib["right_K"]
    img_size = calib.get("image_size", (1280, 720))
    fov_h1, fov_v1 = calculate_fov(K1, img_size[0], img_size[1])
    fov_h2, fov_v2 = calculate_fov(K2, img_size[0], img_size[1])

    summary["fov_left"] = (fov_h1, fov_v1)
    summary["fov_right"] = (fov_h2, fov_v2)

    return summary


def print_compact_summary(dataset_name):
    """Imprime un resumen compacto de la calibración"""
    print(f"\n{'=' * 70}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 70}")

    calib, maps = load_calibration_data(dataset_name)

    # Intrínsecos
    K1 = calib["left_K"]
    K2 = calib["right_K"]
    print(f"\n📷 Intrínsecos:")
    print(
        f"   Focal L: fx={K1[0, 0]:.1f}, fy={K1[1, 1]:.1f} | R: fx={K2[0, 0]:.1f}, fy={K2[1, 1]:.1f}"
    )
    print(f"   Diferencia focal: {abs(K1[0, 0] - K2[0, 0]) / K1[0, 0] * 100:.2f}%")

    # FOV
    img_size = calib.get("image_size", (1280, 720))
    fov_h1, fov_v1 = calculate_fov(K1, img_size[0], img_size[1])
    print(f"   FOV: {fov_h1:.1f}° × {fov_v1:.1f}°")

    # Distorsión
    D1 = calib["left_dist"]
    D2 = calib["right_dist"]
    print(f"\n🔍 Distorsión:")
    print(f"   k1: L={D1[0, 0]:.4f}, R={D2[0, 0]:.4f}")
    print(f"   k2: L={D1[0, 1]:.4f}, R={D2[0, 1]:.4f}")

    # Extrínsecos
    T = calib["T"]
    R = calib["R"]
    baseline = np.linalg.norm(T)
    r = Rot.from_matrix(R)
    euler = r.as_euler("xyz", degrees=True)
    max_angle = np.max(np.abs(euler))

    print(f"\n📐 Extrínsecos:")
    print(f"   Baseline: {baseline:.2f} mm ({baseline / 10:.2f} cm)")
    print(f"   Rotación máx: {max_angle:.2f}°")

    # Rectificación
    P1 = maps["P1"]
    P2 = maps["P2"]
    cy_diff = abs(P1[1, 2] - P2[1, 2])

    print(f"\n✅ Rectificación:")
    print(f"   Alineación epipolar (Δcy): {cy_diff:.4f} px")

    if cy_diff < 0.1:
        print(f"   Estado: ✓ Excelente")
    elif cy_diff < 1.0:
        print(f"   Estado: ⚠ Bueno")
    else:
        print(f"   Estado: ⚠ Revisar")

    # ROI
    roi1 = maps.get("validRoi1")
    if roi1:
        x1, y1, w1, h1 = roi1
        area_pct = (w1 * h1) / (img_size[0] * img_size[1]) * 100
        print(f"   ROI válida: {area_pct:.1f}% de la imagen")
