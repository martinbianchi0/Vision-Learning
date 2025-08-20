# Trabajo Práctico — Umbralización y Análisis de Regiones

**Tema:** Segmentación binaria, morfología y conteo de objetos con OpenCV  
**Estado:** Completado (pipeline reproducible con parámetros ajustables)  
**Formato del repo:** Notebook principal + módulo(s) en `src/` + carpeta de imágenes

---

## 📌 Descripción

Se implementó un pipeline clásico de visión por computadora para segmentar y **contar** objetos en imágenes. El flujo general es:

1) **Preprocesado** (opcional)  
2) **Umbralización** (global, por color en HSV)  
3) **Morfología** (closing/cleaning)  
4) **Componentes conexas** con estadísticas por región  
5) **Filtros** por área/tamaño/posición  
6) **Conteo y anotación** (centroides/overlays para auditoría visual)

El enfoque prioriza **robustez y claridad**: parámetros expuestos tipo *knobs*, funciones modulares y visualizaciones para diagnosticar qué hace cada paso.

---

# 📂 Estructura del Proyecto

```
TP0_UMBRALIZACION_REGIONES.zip
├─ imgs/
│   ├─ inputs/                       # Imágenes de entrada (datasets de prueba)
│   ├─ outputs/                      # Imágenes esperadas de salida
│   └─ resources/                    # Figuras o recursos adicionales
│
├─ src/                              # Módulos por funcionalidad (una sola carpeta)
│  ├─ colorseg.py                 # Segmentación por color (HSV, rangos, máscaras)
│  ├─ marking.py                  # Anotación de resultados (centroides, cruces, overlays)
│  ├─ morphology.py               # Operaciones morfológicas (closing, relleno, limpieza)
│  ├─ utils.py                    # Funciones auxiliares (I/O, paths, helpers varios)
│  └─ visualization.py            # Funciones de visualización y comparación de etapas
│
├─ umbralizacion_regiones.ipynb   # Notebook principal (ejecución y pipeline)
├─ README.txt                     # Documentación del proyecto
└─ requirements.txt               # Dependencias del entorno
```

---

# 📂 Módulos (resumen funcional)

src/
├─ segmentation.py   # Umbral global y segmentación por color (HSV)
├─ morphology.py     # Kernels e interfaces para operaciones morfológicas (closing, cleaning)
├─ regions.py        # Etiquetado de componentes conexas, estadísticas, filtros y conteo
├─ viz.py            # Visualización: comparadores de etapas, overlays y helpers de gráficos
└─ utils.py          # Utilidades generales: I/O, paths y funciones auxiliares

# Nota:
# Se mantuvo una sola carpeta `src/` con módulos organizados por funcionalidad,
# en lugar de crear un archivo separado por cada función.

---

## 📊 Contenido del trabajo

### 1️⃣ Umbralización / Binarización
- **Global**
- **Por color (HSV)** cuando el canal **H** o combinaciones **H–S–V** separan mejor objeto/fondo.

### 2️⃣ Morfología
- Kernels elípticos/cuadrados vía `cv2.getStructuringElement`.
- **Closing** para cerrar huecos y consolidar regiones.
- Limpieza de ruido fino y bordes irregulares.

### 3️⃣ Componentes Conexas y Métricas
- Etiquetado con `cv2.connectedComponentsWithStats`.
- Extracción de **área, bbox, centroides**.
- Filtros por **área mínima/máxima** y descarte de regiones marginales.

### 4️⃣ Conteo y Anotación
- Conteo estable tras filtros.
- **Overlays** sobre la imagen original para validar rápido (puntos en centroides, cajas/bordes si corresponde).
- Utilidades de plot para comparar **etapas** (entrada → máscara → post-morfología → anotación).

### 5️⃣ Parámetros ajustables (knobs)
- Modo de umbralización (global/Otsu/adaptativa/HSV).
- `blockSize` y `C` (adaptativa).
- Tamaño/forma del **kernel** morfológico e **iteraciones**.
- Rangos HSV si se segmenta por color.
- Filtros de **área** y descarte en bordes.

### 6️⃣ Resultados y límites
- **Máscaras** limpias y conteo reproducible en escenas con ruido moderado.
- En casos con **superposición fuerte** o **sombras severas**, se sugiere sumar filtros de **forma** (p.ej., circularidad) o técnicas más avanzadas.

---

## 🛠 Dependencias

- Python 3.8+
- `opencv-python`
- `numpy`
- `matplotlib`
- (opcional) Jupyter / Google Colab

Instalación rápida:
```bash
pip install opencv-python numpy matplotlib