# Referencia — Sierra RECHAZADA (no afilable)

> Contraparte del caso ideal ([LU3F-0300.md](LU3F-0300.md)). Define cómo se ve una sierra que
> el operario **descarta**: no se puede reafilar y debe salir de circulación.

## Identificación

| Dato | Valor |
|---|---|
| Marca a mano en el cuerpo | "DEC PALOMAR" · "N1" (identificación interna del taller) |
| Tipo | Sierra circular con dientes de metal duro, ya muy usada |
| Estado general | Cuerpo oxidado (pátina marrón), puntas con óxido |
| Veredicto del operario | **No afilable** — diente faltante |

## Por qué se rechaza

**Motivo principal (dictaminado por el operario):**
- **Diente faltante.** Falta al menos un diente / inserto. Un diente menos rompe el paso
  regular del corte y no se recupera afilando: hay que descartar la sierra.

**Condiciones visibles que acompañan (secundarias):**
- **Óxido** generalizado en el cuerpo y en las puntas de los dientes.
- Desgaste avanzado propio de una herramienta al final de su vida útil.

> Distinción clave para el modelo: **desgastado ≠ no afilable**. Una sierra *desgastada* se
> reafila y vuelve a servir; una con *diente faltante* o *fisura* se descarta. El sistema tiene
> que aprender a separar "reafilable" de "rechazo definitivo", no solo "sana vs fea".

## Resultado de esta prueba

- **Curado por nitidez:** de 51 fotos, ~35 salieron movidas y las descartó el filtro del
  sistema (Laplaciano < 60). Quedaron 16 cenitales nítidas → deduplicadas a 6 representativas.
- **Detección de disco:** correcta (Hough localiza la sierra).
- **Autolocalización del diente faltante:** se intentó por desenrollado polar de la corona de
  dientes y **falló** (contó 0–5 dientes donde hay ~90). Con foto de mano, luz de taller y
  óxido, el método clásico no es fiable. Es exactamente el problema que resuelve YOLO entrenado:
  por eso el diente faltante se marca a mano en Roboflow, no se inventa una caja automática.

## Contenido

```
referencia/
  R01-no-afilable.md                  <- este documento
  rechazada_R01_disco.jpg             <- anotación a nivel de disco (rojo = rechazada)
  rechazadas/R01-no-afilable/
    cenital_01..06.jpg                <- cenitales nítidas, deduplicadas
```

## Esquema binario: ya está etiquetada

Con el esquema **binario** (`aprobada` / `rechazada`) esta sierra queda etiquetada a nivel de
disco completo como `rechazada` (clase 1) — no hace falta marcar a mano el diente faltante, la
sierra entera se descarta. Las 6 cenitales están sembradas en `data/dataset/crudo/` con esa
etiqueta.

> Si en el futuro se pasa al esquema por defecto específico (para que el sistema diga *por qué*
> rechaza), ahí sí habrá que dibujar en Roboflow la caja del `diente_faltante` sobre estas
> mismas fotos. Por eso se conservan en la referencia.
