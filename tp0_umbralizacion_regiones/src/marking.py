from src.morphology import separar_componentes, reconstruir_confite, bbox_y_centro, _detect_points_with_doublets
import numpy as np
import cv2

def anotar_overlay(COLORS_BGR, overlay_bgr, masks_col, min_area=100):
    """
    Dibuja bounding boxes, cruces y etiquetas por color.

    Parámetros:
      - COLORS_BGR: Diccionario {clase: color BGR}.
      - overlay_bgr: Imagen base sobre la que se dibuja.
      - masks_col : Máscaras binarias por color.
      - min_area  : Área mínima de componentes.

    Returns:
      - out    : Imagen anotada.
      - conteos: Dict con cantidad por color.
    """
    out = overlay_bgr.copy()
    orden = ["Naranja", "Amarillo", "Verde", "Azul", "Rojo"]
    conteos = {c: 0 for c in orden}

    Himg = out.shape[0]
    font_scale = max(0.55, 0.0007 * Himg)
    thick      = max(1,     int(0.0022 * Himg))

    # iteraciones de erosión por color (tus knobs)
    iters_por_color = {
        "Rojo": 8,
        "Naranja": 20,
        "Amarillo": 1,
        "Verde": 1,
        "Azul": 1,
    }

    for c in orden:
        if c not in masks_col:
            continue

        color = COLORS_BGR.get(c, (255,255,255))  # fallback blanco
        m_orig = (masks_col[c] > 0).astype(np.uint8) * 255
        m_orig = cv2.medianBlur(m_orig, 3)

        comps, labels = separar_componentes(
            m_orig, iters=iters_por_color.get(c,1), min_area=min_area
        )

        for idx, lbl in enumerate(comps, start=1):
            comp_mask  = (labels == lbl).astype(np.uint8) * 255
            comp_final = reconstruir_confite(
                comp_mask, m_orig, iters=iters_por_color.get(c,1)
            )

            bbox = bbox_y_centro(comp_final)
            if bbox is None:
                continue

            x, y, w, h, cx, cy = bbox
            conteos[c] += 1

            # Caja del color correspondiente
            cv2.rectangle(out, (x, y), (x+w, y+h), color, 2)

            # Cruz del color correspondiente
            cv2.drawMarker(
                out, (cx, cy), color,
                markerType=cv2.MARKER_CROSS,
                markerSize=max(10, int(0.14 * min(w, h))),
                thickness=2
            )

            # Texto del color correspondiente
            cv2.putText(
                out, f"{c} {idx}", (x, max(0, y-5)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thick, cv2.LINE_AA
            )

    return out, conteos

def _draw_crosses(img_bgr, detections,
                  color_single=(0,0,255),  # rojo
                  color_double=(0,255,0),  # verde
                  size=12, thickness=2):
    """
    Dibuja cruces sobre detecciones.

    Parámetros:
      - img_bgr     : Imagen original.
      - detections  : Lista de dicts con 'pt' y flag 'from_doublet'.
      - color_single: Color de cruces simples.
      - color_double: Color de cruces en dobletes.
      - size, thickness: Tamaño y grosor de cruces.

    Returns:
      - out: Imagen con cruces.
    """
    out = img_bgr.copy()
    for det in detections:
        x, y = det['pt']
        col = color_double if det.get('from_doublet', False) else color_single
        cv2.drawMarker(out, (int(x), int(y)), col, cv2.MARKER_CROSS, size, thickness)
    return out

def annotate_components_crosses(img_bgr, mask_bin, area_min=200, area_max=6000,min_w=5, min_h=5,
                                # --- knobs para detectar/partir dobletes ---
                                enable_doublets=True,
                                dt_min=5,            # px de "radio" mínimo en picos del distance transform
                                peak_min_sep=20,     # px de separación mínima entre dos picos
                                circ_min_each=0.50,  # circularidad mínima por lóbulo (4πA/P^2)
                                post_close_ksize=3   # closing chiquito tras el split
                                ):
    """
    Marca centroides con cruces, corrigiendo dobletes.

    Parámetros:
      - img_bgr : Imagen original BGR.
      - mask_bin: Máscara binaria.
      - area_min/area_max: Filtro de área px².
      - min_w/min_h: Filtro de tamaño mínimo.
      - enable_doublets: Si habilita corrección de dobletes.
      - dt_min, peak_min_sep, circ_min_each, post_close_ksize: Knobs de split.

    Returns:
      - out  : Imagen con cruces.
      - count: Número de células detectadas.
    """
    detections = _detect_points_with_doublets(
        mask_bin=mask_bin,
        area_min=area_min, area_max=area_max,
        min_w=min_w, min_h=min_h,
        enable_doublets=enable_doublets,
        dt_min=dt_min, peak_min_sep=peak_min_sep,
        circ_min_each=circ_min_each, post_close_ksize=post_close_ksize
    )
    out = _draw_crosses(img_bgr, detections,
                        color_single=(0,0,255),
                        color_double=(0,0,255),
                        size=12, thickness=2)
    return out, len(detections)
