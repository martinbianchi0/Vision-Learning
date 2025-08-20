import numpy as np
import cv2

def rango_color(h_low, h_high, s_min=60, v_min=50, v_max=255):
    """
    Devuelve límites HSV para un rango de color.

    Parámetros:
      - h_low, h_high: Hue mínimo y máximo.
      - s_min: Saturación mínima.
      - v_min, v_max: Valores mínimo y máximo.

    Returns:
      - (lo, hi): Tupla de arrays HSV con límites inferior y superior.
    """
    lo = np.array([h_low, s_min, v_min], np.uint8)
    hi = np.array([h_high, 255,  v_max], np.uint8)
    return lo, hi

def computar_mascaras(hsv, RANGOS, sat_min=120, val_min=45, val_max=245):
    """
    Genera máscaras binarias para distintos colores en un espacio HSV.

    Parámetros:
      - hsv: Imagen en espacio HSV.
      - RANGOS: Diccionario con rangos de H por color.
      - sat_min, val_min, val_max: Umbrales globales de S/V.

    Returns:
      - mask_sv   : Máscara global por S/V.
      - masks_col : Diccionario con máscaras por color.
    """
    # pre-máscara por S/V (limpia grises, sombras y reflejos fuertes)
    mask_sv = cv2.inRange(hsv, (0, sat_min, val_min), (179, 255, val_max))

    # colores (cada máscara AND con S/V)
    m_rojo = cv2.inRange(hsv, *RANGOS["Rojo_1"]) | cv2.inRange(hsv, *RANGOS["Rojo_2"])
    m_rojo = cv2.bitwise_and(m_rojo, mask_sv)

    masks_col = {
        "Rojo":     m_rojo,
        "Naranja":  cv2.bitwise_and(cv2.inRange(hsv, *RANGOS["Naranja"]),  mask_sv),
        "Amarillo": cv2.bitwise_and(cv2.inRange(hsv, *RANGOS["Amarillo"]), mask_sv),
        "Verde":    cv2.bitwise_and(cv2.inRange(hsv, *RANGOS["Verde"]),    mask_sv),
        "Azul":     cv2.bitwise_and(cv2.inRange(hsv, *RANGOS["Azul"]),     mask_sv),
    }
    return mask_sv, masks_col

def apply_otsu(gray_img):
    """
    Aplica el método de Otsu a una imagen en grises.

    Parámetros:
      - gray_img: Imagen en escala de grises.

    Returns:
      - mask      : Máscara binaria resultante.
      - thresh_val: Umbral calculado por Otsu.
    """
    thresh_val, mask = cv2.threshold(
        gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask, thresh_val

def color_mask(hsv, sat_min, val_min, val_max,
               h_red1, h_red2,
               h_bp_lo, h_bp_hi,
               bp_smin, bp_vmin):
    """
    Máscara HSV: incluye rojo y excluye azul/morado.

    Parámetros:
      - hsv: Imagen en HSV.
      - sat_min, val_min, val_max: Umbrales globales S/V.
      - h_red1, h_red2: Rangos de H para rojo.
      - h_bp_lo, h_bp_hi: Rangos de H a excluir (azul/morado).
      - bp_smin, bp_vmin: Umbrales S/V para la exclusión.

    Returns:
      - mask : Máscara final limpia.
      - m_red: Máscara solo de rojo.
    """
    # 1) S/V global
    mask_sv = cv2.inRange(hsv, (0, sat_min, val_min), (179, 255, val_max))

    # 2) Rojo (dos rangos)
    m_r1 = cv2.inRange(hsv, (h_red1[0], sat_min, val_min), (h_red1[1], 255, val_max))
    m_r2 = cv2.inRange(hsv, (h_red2[0], sat_min, val_min), (h_red2[1], 255, val_max))
    m_red = cv2.bitwise_or(m_r1, m_r2)

    # 3) Excluir azul/morado
    m_bp = cv2.inRange(hsv, (h_bp_lo, bp_smin, bp_vmin), (h_bp_hi, 255, 255))

    # 4) Final
    mask = cv2.bitwise_and(m_red, mask_sv)
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(m_bp))

    return mask, m_red 