import numpy as np
import cv2

def separar_componentes(mask, iters=1, min_area=100):
    """
    Separa componentes conectadas en una máscara binaria.

    Parámetros:
      - mask: Máscara binaria de entrada (np.uint8).
      - iters: Iteraciones de erosión para separar objetos pegados.
      - min_area: Área mínima para aceptar un componente.

    Returns:
      - comps: Lista de labels de componentes válidos.
      - labels: Imagen con etiquetas de cada píxel.
    """
    kernel = np.ones((3,3), np.uint8)
    m_sep = cv2.erode(mask, kernel, iterations=iters)
    n, labels, stats, cent = cv2.connectedComponentsWithStats(m_sep, 8, cv2.CV_32S)
    comps = [lbl for lbl in range(1, n) if stats[lbl, cv2.CC_STAT_AREA] >= min_area]
    comps.sort(key=lambda lbl: (cent[lbl][1], cent[lbl][0]))  # orden de lectura
    return comps, labels

def reconstruir_confite(comp_mask, m_orig, iters):
    """
    Reconstruye el tamaño real de un confite separado.

    Parámetros:
      - comp_mask: Máscara binaria de un confite erosionado.
      - m_orig: Máscara original del color correspondiente.
      - iters: Iteraciones de dilatación para volver al tamaño real.

    Returns:
      - Máscara binaria del confite reconstruido.
    """
    kernel = np.ones((3,3), np.uint8)
    comp_grow = cv2.dilate(comp_mask, kernel, iterations=iters)
    return cv2.bitwise_and(comp_grow, m_orig)

def bbox_y_centro(comp_mask):
    """
    Calcula bounding box y centroide de un componente.

    Parámetros:
      - comp_mask: Máscara binaria del componente.

    Returns:
      - (x,y,w,h,cx,cy): Coordenadas del bounding box y centroide.
        Devuelve None si no hay contornos.
    """
    cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cmax = max(cnts, key=cv2.contourArea)
    x,y,w,h = cv2.boundingRect(cmax)
    M = cv2.moments(cmax)
    cx, cy = (int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])) if M["m00"]>0 else (x+w//2,y+h//2)
    return x,y,w,h,cx,cy

def fill_holes(mask):
    """
    Rellena agujeros internos sin expandir bordes.

    Parámetros:
      - mask: Máscara binaria 0/255.

    Returns:
      - Máscara con cavidades internas rellenadas.
    """
    m = mask.copy()
    h, w = m.shape[:2]
    ff = m.copy()
    aux = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(ff, aux, (0,0), 255)    # fondo conectado al borde
    holes = cv2.bitwise_not(ff)           # cavidades internas
    return cv2.bitwise_or(m, holes)

def remove_small_border_components(mask_bin, 
                                   alpha=0.60, 
                                   only_corners=False,
                                   area_min_ref=200, area_max_ref=6000,
                                   min_ref=8):
    """
    Elimina componentes de borde cuya área es 'chica' en relación al tamaño típico interior.
    
    Parámetros:
      - mask_bin     : máscara binaria 0/255.
      - alpha        : factor sobre la mediana interior. thr = alpha * mediana_area_interior.
      - only_corners : si True, filtra SOLO los que tocan 2+ bordes (esquinas).
                       si False, filtra cualquiera que toque >=1 borde (laterales o esquinas).
      - area_min_ref / area_max_ref : rango de área para estimar el tamaño típico (evita ruido/outliers).
      - min_ref      : mínimo de muestras interiores para una mediana confiable; si no, cae a fallback.

    Returns:
      - keep_mask : máscara binaria 0/255 sin las células pequeñas de borde.
      - thr       : umbral de área usado (float), por si querés loguear.
      - med       : mediana de área interior (float) usada como referencia.
    """
    m = (mask_bin > 0).astype(np.uint8) * 255
    H, W = m.shape[:2]
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, 8, cv2.CV_32S)

    # 1) estimar tamaño típico con interiores (no tocan bordes) dentro de rango razonable
    interior_areas = []
    for lbl in range(1, n):
        x, y, w, h, A = stats[lbl, 0], stats[lbl, 1], stats[lbl, 2], stats[lbl, 3], stats[lbl, cv2.CC_STAT_AREA]
        if A < area_min_ref or A > area_max_ref:
            continue
        touches = (x == 0) + (x + w >= W) + (y == 0) + (y + h >= H)
        if touches == 0:
            interior_areas.append(A)

    if len(interior_areas) >= min_ref:
        med = float(np.median(interior_areas))
    else:
        # Fallback: mediana global en rango, aunque toquen borde (menos robusto)
        pool = [float(stats[lbl, cv2.CC_STAT_AREA]) for lbl in range(1, n)
                if area_min_ref <= stats[lbl, cv2.CC_STAT_AREA] <= area_max_ref]
        med = float(np.median(pool)) if len(pool) > 0 else float(area_min_ref * 1.5)

    thr = alpha * med

    # 2) construir máscara filtrando "borde y chico"
    keep = np.zeros_like(m)
    for lbl in range(1, n):
        x, y, w, h, A = stats[lbl, 0], stats[lbl, 1], stats[lbl, 2], stats[lbl, 3], stats[lbl, cv2.CC_STAT_AREA]
        roi = (labels[y:y+h, x:x+w] == lbl).astype(np.uint8) * 255

        touch_left   = (x == 0)
        touch_right  = (x + w >= W)
        touch_top    = (y == 0)
        touch_bottom = (y + h >= H)
        touches = int(touch_left) + int(touch_right) + int(touch_top) + int(touch_bottom)

        is_border = (touches >= 2) if only_corners else (touches >= 1)

        # Regla: si está en el borde y su área es menor al umbral => descartar
        if is_border and (A < thr):
            continue

        keep[y:y+h, x:x+w] = cv2.bitwise_or(keep[y:y+h, x:x+w], roi)

    return keep, thr, med

def _circularity(bin_mask):
    """
    Circularidad 4πA/P² de la mayor componente.

    Parámetros:
      - bin_mask: Máscara binaria.

    Returns:
      - (c, A, P): Circularidad, área y perímetro.
    """
    cnts, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return 0.0, 0.0, 0.0
    c = max(cnts, key=cv2.contourArea)
    A = float(cv2.contourArea(c))
    P = float(cv2.arcLength(c, True))
    if P <= 1e-6:
        return 0.0, A, P
    return 4.0 * np.pi * A / (P * P), A, P

def _local_max_dt(dt, dt_min_val=5, suppress_radius=6, max_peaks=3):
    """"
    Picos locales del distance transform con NMS.

    Parámetros:
      - dt: Distance transform (float32).
      - dt_min_val: Valor mínimo para considerar pico.
      - suppress_radius: Radio de supresión (NMS).
      - max_peaks: Máximo de picos a devolver.

    Returns:
      - Lista [(y, x, v), ...] de picos.
    """
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    dt_dil = cv2.dilate(dt, k)
    mask_max = (dt == dt_dil) & (dt >= dt_min_val)
    ys, xs = np.where(mask_max)
    vals = dt[ys, xs]
    if len(vals) == 0:
        return []
    idx = np.argsort(-vals)
    peaks = []
    for i in idx:
        y, x, v = int(ys[i]), int(xs[i]), float(vals[i])
        # supresión por radio
        too_close = False
        for (py, px, pv) in peaks:
            if (x - px)**2 + (y - py)**2 <= suppress_radius**2:
                too_close = True
                break
        if not too_close:
            peaks.append((y, x, v))
        if len(peaks) >= max_peaks:
            break
    return peaks

def _split_by_two_seeds(roi_mask, p1, p2, close_ksize=3):
    """
    Divide un ROI en dos regiones (Voronoi a 2 semillas).

    Parámetros:
      - roi_mask: Máscara del ROI (0/255).
      - p1, p2  : Semillas (y, x).
      - close_ksize: Closing opcional para limpieza.

    Returns:
      - (roi1, roi2): Máscaras binarias resultantes (o (None, None)).
    """
    ys, xs = np.where(roi_mask > 0)
    if len(ys) == 0:
        return None, None
    d1 = (xs - p1[1])**2 + (ys - p1[0])**2
    d2 = (xs - p2[1])**2 + (ys - p2[0])**2
    assign1 = d1 <= d2
    roi1 = np.zeros_like(roi_mask, dtype=np.uint8); roi1[ys[assign1], xs[assign1]] = 255
    roi2 = np.zeros_like(roi_mask, dtype=np.uint8); roi2[ys[~assign1], xs[~assign1]] = 255

    # limpieza suave
    if close_ksize and close_ksize >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        roi1 = cv2.morphologyEx(roi1, cv2.MORPH_CLOSE, ker, iterations=1)
        roi2 = cv2.morphologyEx(roi2, cv2.MORPH_CLOSE, ker, iterations=1)

    # quedarnos con la mayor CC de cada lado
    def _largest_cc(b):
        n, labs, stats, _ = cv2.connectedComponentsWithStats((b>0).astype(np.uint8)*255, 8, cv2.CV_32S)
        if n <= 1:
            return np.zeros_like(b, dtype=np.uint8)
        lbl = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return (labs == lbl).astype(np.uint8) * 255

    roi1 = _largest_cc(cv2.bitwise_and(roi1, roi_mask))
    roi2 = _largest_cc(cv2.bitwise_and(roi2, roi_mask))
    return roi1, roi2

def _centroid(bin_mask):
    """
    Centroide (cx, cy) de una máscara.

    Parámetros:
      - bin_mask: Máscara binaria.

    Returns:
      - (cx, cy) o None si no hay masa.
    """
    m = cv2.moments(bin_mask, binaryImage=True)
    if abs(m["m00"]) < 1e-6:
        return None
    return (int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"]))

def _detect_points_with_doublets(mask_bin,
                                 area_min, area_max,
                                 min_w, min_h,
                                 enable_doublets,
                                 dt_min, peak_min_sep,
                                 circ_min_each, post_close_ksize):
    """
    Detecta centroides y corrige dobletes.

    Parámetros:
      - mask_bin: Máscara 0/255.
      - area_min/area_max: Filtro de área.
      - min_w/min_h: Filtro de tamaño.
      - enable_doublets: Habilita split por picos en DT.
      - dt_min, peak_min_sep, circ_min_each, post_close_ksize: Knobs del split.

    Returns:
      - Lista de dicts: {'pt': (x,y), 'from_doublet': bool}.
    """
    m = (mask_bin > 0).astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(m, 8, cv2.CV_32S)

    H, W = m.shape[:2]
    pad = 3
    detections = []

    for lbl in range(1, n):
        x, y, w, h, A = stats[lbl, 0], stats[lbl, 1], stats[lbl, 2], stats[lbl, 3], stats[lbl, cv2.CC_STAT_AREA]
        if A < area_min or A > area_max:
            continue
        if w < min_w or h < min_h:
            continue

        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(W, x + w + pad); y1 = min(H, y + h + pad)
        roi_mask = ((labels[y0:y1, x0:x1] == lbl).astype(np.uint8) * 255)

        added = False
        if enable_doublets:
            dt = cv2.distanceTransform((roi_mask>0).astype(np.uint8), cv2.DIST_L2, 3)
            peaks = _local_max_dt(dt, dt_min_val=dt_min, suppress_radius=5, max_peaks=3)
            if len(peaks) >= 2:
                (py1, px1, _), (py2, px2, _) = sorted(peaks, key=lambda p: -p[2])[:2]
                sep = np.hypot(px1 - px2, py1 - py2)
                if sep >= peak_min_sep:
                    roi1, roi2 = _split_by_two_seeds(roi_mask, (py1, px1), (py2, px2), close_ksize=post_close_ksize)
                    if roi1 is not None and roi2 is not None:
                        c1, A1, _ = _circularity(roi1)
                        c2, A2, _ = _circularity(roi2)
                        if c1 >= circ_min_each and c2 >= circ_min_each and A1 >= area_min/2 and A2 >= area_min/2:
                            p1 = _centroid(roi1); p2 = _centroid(roi2)
                            if p1 is not None and p2 is not None:
                                detections.append({'pt': (x0 + p1[0], y0 + p1[1]), 'from_doublet': True})
                                detections.append({'pt': (x0 + p2[0], y0 + p2[1]), 'from_doublet': True})
                                added = True

        if not added:
            cx, cy = int(centroids[lbl][0]), int(centroids[lbl][1])
            detections.append({'pt': (cx, cy), 'from_doublet': False})

    return detections