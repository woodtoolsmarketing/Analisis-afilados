"""Interfaces de linea de comandos del sistema de analisis de afilado.

Contiene los tres puntos de entrada del proyecto:
  - run_live: inspeccion en vivo con la webcam y bucle de feedback del operario.
  - train: entrenamiento del modelo YOLO sobre el dataset corregido.
  - prepare_dataset: extraccion de frames, division train/val y fusion del feedback.

Este paquete se mantiene vacio a proposito: cada CLI se importa por separado
para que el paquete raiz no arrastre dependencias pesadas.
"""
