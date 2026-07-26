import time
import matplotlib.pyplot as plt
import numpy as np


def normalize_disparity(disp: np.ndarray) -> np.ndarray:
    """
    Normaliza un mapa de disparidad a rango 0-255 para visualización.

    Args:
        disp: Disparidad en float/entero con posibles NaN.

    Returns:
        Imagen uint8 normalizada en [0, 255].
    """
    disp = disp.astype(np.float32)
    m, M = np.nanmin(disp), np.nanmax(disp)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return np.zeros_like(disp, dtype=np.uint8)
    dvis = 255 * (disp - m) / (M - m)
    return dvis.astype(np.uint8)


def show_disparity(title: str, disp: np.ndarray, cmap: str = 'magma'):
    """
    Muestra un mapa de disparidad con matplotlib y barra de color.

    Args:
        title: Título de la figura.
        disp: Mapa de disparidad 2D.
        cmap: Colormap de matplotlib.

    Returns:
        None.
    """
    plt.figure(figsize=(10,4))
    plt.title(title)
    plt.imshow(disp, cmap=cmap)
    plt.colorbar(); plt.axis('off')
    plt.show()


def time_call(fn, *args, **kwargs):
    """
    Cronometra una llamada a función y devuelve su salida y tiempo en ms.

    Args:
        fn: Función a ejecutar.
        *args: Argumentos posicionales.
        **kwargs: Argumentos nombrados.

    Returns:
        Tuple (resultado, milisegundos)
    """
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = (time.perf_counter() - t0) * 1000.0
    return out, dt