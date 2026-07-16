"""Configuracion comun de pytest.

El proyecto no se instala como paquete: los modulos viven en src/. Sin este ajuste
del sys.path los tests no encontrarian 'afilado' y fallarian con ImportError antes
de ejecutar una sola asercion.
"""

from __future__ import annotations

import sys
from pathlib import Path

RAIZ_REPO = Path(__file__).resolve().parents[1]
DIRECTORIO_FUENTE = RAIZ_REPO / "src"

if str(DIRECTORIO_FUENTE) not in sys.path:
    sys.path.insert(0, str(DIRECTORIO_FUENTE))
