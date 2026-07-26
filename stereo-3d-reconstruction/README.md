## Notebooks y orden de ejecución

- 00_rectificacion.ipynb: Rectifica pares estéreo y guarda imágenes rectificadas y stereo_calibration.json.
- 01_disparidad.ipynb: Calcula disparidad en las imágenes rectificadas y ajusta parámetros (vista previa/diagnóstico).
- 02_reconstruction_buda.ipynb / 02_reconstruction_tambor.ipynb: Reconstrucción completa del objeto, se proyecta a 3D con Q, estima poses con Charuco y guarda points.npy y poses.npy, además de las nubes de puntos del dataset correspondiente.
- 03_mesh_buda.ipynb / 03_mesh_tambor.ipynb (o mesh.ipynb / mesh_tambor.ipynb): Genera malla desde la nube (Poisson) y exporta .ply.



## Organización de `datasets/`

- datasets/<dataset_name>/
  - cfg/: configuraciones/presets de calibración y rectificación.
  - data/
    - calib/: imágenes para calibrar (parejas left/right).
    - captures/: capturas crudas para el pipeline.
    - rect/: imágenes rectificadas y `stereo_calibration.json`.
    - results/: salidas del pipeline (`points.npy`, `poses.npy`, nubes/meshes .ply).
    - stereo_calibration.pkl, stereo_maps.pkl: parámetros y mapas de rectificación.
