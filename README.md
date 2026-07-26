# Visión por computadora clásica — tres pipelines, con la geometría escrita a mano

Tres trabajos de visión artificial de la materia **Visión Artificial (I308)** — Universidad de San Andrés, 2do semestre 2025. Nota de la cursada: 9.5/10.

**Lo que distingue este repo:** la estimación de homografías está **implementada desde cero** —DLT armando la matriz del sistema y resolviendo por SVD, más un RANSAC propio con umbral, tope de iteraciones y nivel de confianza—. **No se usa `cv2.findHomography`.** De OpenCV se usan el detector de features y el warping; la geometría es propia. Lo mismo aplica a la separación de objetos pegados por transformada de distancia y al pipeline estéreo completo.

## Contenido

| Carpeta | Qué hace | Lo técnicamente interesante |
|---|---|---|
| [`panoramas-homography/`](panoramas-homography/) | Panorámicas a partir de tripletes de fotos | **Homografía por DLT (SVD) y RANSAC propios**; keypoints SIFT + ANMS para distribución uniforme; matching por regla de Lowe con verificación cruzada; cálculo del lienzo óptimo; blending por transformada de distancia |
| [`segmentation-and-counting/`](segmentation-and-counting/) | Segmentación y conteo de objetos | Umbralización global y por color en HSV, Otsu, morfología, componentes conexas con estadísticas por región; **separación de objetos pegados por máximos locales de la transformada de distancia** y medida de circularidad |
| [`stereo-3d-reconstruction/`](stereo-3d-reconstruction/) | Reconstrucción 3D a partir de pares estéreo | Calibración y rectificación estéreo, mapa de disparidad (**tres métodos comparados, con tiempos**), reproyección con la matriz Q, pose por **PnP** sobre tablero ChArUco, acumulación de vistas para cobertura de 360°, segmentación por Oriented Bounding Box y malla por **Poisson** exportada a `.ply` |

Cada carpeta tiene su propio README con el detalle, y las dos primeras incluyen el informe entregado.

## Resultados

- **Panorámicas:** reconstrucción continua sobre tres conjuntos —campus de UdeSA, un cuadro y un set propio de árboles tomado en la vía pública—. El caso del cuadro es el interesante: con objetos 3D en escena (una mesa) el plano del cuadro se reconstruye bien y el resto no, que es exactamente lo que predice el modelo de homografía.
- **Reconstrucción 3D:** la calibración estéreo devolvió una línea base de **≈ 60 mm** sobre el eje horizontal con rotación cercana a la identidad, lo que valida la geometría del par. Altura máxima de las reconstrucciones: **buda 276 mm, tambor 175 mm**, medidas sobre la nube segmentada y filtrada. Son mediciones del modelo reconstruido: **no hay una medición del objeto real publicada contra la cual compararlas**, así que describen la reconstrucción, no su exactitud.
- **Conteo de objetos:** validado visualmente con overlays de centroides sobre imágenes de células sanguíneas y de confites, que es la forma de auditoría que pedía la consigna.

## Autoría

- `panoramas-homography/` — Martín Bianchi y Federico Gutman (informe firmado, septiembre 2025).
- `stereo-3d-reconstruction/` — Joaquín León, Martín Bianchi, Federico Gutman y Juan Andrés Quiroga (informe firmado, octubre 2025).
- `segmentation-and-counting/` — no declara autores en la entrega.

## Nota sobre este repositorio

Reúne lo que antes estaban dos repos separados (`Vision-Learning` y `Vision-Learning-3D`). El segundo quedó archivado. Es **coursework**, y está bien dicho así: son trabajos de cursada, no productos. El proyecto integrador de esta misma materia es otro y vive aparte: [VisionTrafficGuard](https://github.com/martinbianchi0/VisionTrafficGuard), un sistema de fiscalización de tránsito por visión.
