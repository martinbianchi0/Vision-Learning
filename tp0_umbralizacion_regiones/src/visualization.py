import matplotlib.pyplot as plt
import cv2

def plot_hue_histograms(hsvs, bins=180, titles=None):
    """
    Grafica en una figura los histogramas del canal H de tres imágenes HSV.

    Parámetros:
      - hsv1, hsv2, hsv3: imágenes en espacio HSV (np.ndarray).
      - bins: cantidad de bins del histograma (default=180, uno por cada valor de H).
      - titles: lista de títulos opcionales para los subplots.

    Returns:
      - None (muestra la figura con los 3 histogramas).
    """
    n = len(hsvs)

    plt.figure(figsize=(5*n, 4))
    for i, hsv in enumerate(hsvs, 1):
        h_channel = hsv[:, :, 0]
        hist = cv2.calcHist([h_channel], [0], None, [bins], [0, bins])
        plt.subplot(1, n, i)
        plt.plot(hist, color='r')
        plt.xlim([0, bins])
        plt.xlabel("Hue")
        plt.ylabel("Frecuencia")
        if titles:
            plt.title(titles[i-1])
        else:
            plt.title(f"Imagen {i}")
    plt.tight_layout()
    plt.show()

def show_row(titles, imgs):
    """
    Muestra varias imágenes en una fila con sus títulos.

    Parámetros:
      - titles: Lista de títulos.
      - imgs  : Lista de imágenes (grises o BGR).
    """
    n = len(imgs)
    plt.figure(figsize=(6*n, 6))
    for i, (t, im) in enumerate(zip(titles, imgs), 1):
        plt.subplot(1, n, i)
        if im.ndim == 2:  # imagen en escala de grises
            plt.imshow(im, cmap='gray')
        else:  # imagen a color
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        plt.title(t)
        plt.axis('off')
    plt.show()

def plot_colores_por_imagen(img, masks_col, titulo="imagen"):
    """
    Muestra máscaras y overlays por color en 2 filas.

    Parámetros:
      - img      : Imagen BGR.
      - masks_col: Diccionario de máscaras binarias por color.
      - titulo   : Texto base para títulos de subplots.
    """
    colores = ["Rojo", "Naranja", "Amarillo", "Verde", "Azul"]
    n = len(colores)
    plt.figure(figsize=(4*n, 8))

    # fila de máscaras
    for i, c in enumerate(colores, start=1):
        plt.subplot(2, n, i)
        plt.imshow(masks_col[c], cmap='gray')
        plt.title(f"{titulo} - {c} (mask)")
        plt.axis('off')

    # fila de overlays
    for i, c in enumerate(colores, start=1):
        overlay = cv2.bitwise_and(img, img, mask=masks_col[c])
        plt.subplot(2, n, n + i)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title(f"{titulo} - {c} (overlay)")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_otsu(gray_img, mask, thresh_val, title="Imagen"):
    """
    Grafica histograma con umbral Otsu y su máscara binaria.

    Parámetros:
      - gray_img  : Imagen en escala de grises.
      - mask      : Máscara binaria resultante.
      - thresh_val: Umbral Otsu usado.
      - title     : Texto para los títulos.
    """
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    plt.figure(figsize=(12,4))

    # Histograma + línea
    plt.subplot(1, 2, 1)
    plt.plot(hist, color="black")
    plt.axvline(x=thresh_val, color="red", linestyle="--", label=f"Otsu = {thresh_val:.0f}")
    plt.title(f"Histograma + Otsu ({title})")
    plt.xlabel("Intensidad (0–255)")
    plt.ylabel("Frecuencia")
    plt.legend()

    # Máscara binaria
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Máscara binaria ({title})")
    plt.axis("off")

    plt.show()
