# Auditoría del catálogo de capacidades P2

## Fuente y procedencia

- Página oficial consultada: [Datos abiertos MiBici](https://www.mibici.net/es/datos-abiertos/).
- Catálogo oficial utilizado provisionalmente: [`nomenclatura_2026_09.csv`](https://www.mibici.net/site/assets/files/1118/nomenclatura_2026_09.csv).
- Versión del catálogo: septiembre de 2026.
- SHA-256: `206e45fe633922b634d858142fa21e6fd9f4c421b6435c02980ef290cc2df90d`.
- El CSV y su JSON de procedencia se guardan en `data/external/`, ruta excluida de Git.
- Reconstrucción y auditoría: `python -m mibici.modeling.p2_catalog`.

No se encontró una nomenclatura histórica de estaciones de 2025 enlazada en la
página oficial consultada. El catálogo de 2026-09 se usa solo como referencia
provisional. **No se asume que sus capacidades o estaciones fueran las mismas
durante 2025.**

## Cobertura de enero a octubre de 2025

La relación se calculó entre `id` del catálogo y `station_id` del Parquet
completado. Para evitar consultar el periodo reservado, solo se leyeron filas
con `2025-01-01 <= hour < 2025-11-01`.

| Medida | Resultado |
|---|---:|
| Filas del catálogo | 484 |
| IDs únicos del catálogo | 484 |
| Estaciones con registros entre enero y octubre | 370 |
| Estaciones con coincidencia | 370 |
| Estaciones sin coincidencia | 0 |
| Cobertura de estaciones | 100,00 % |
| Registros estación-hora del periodo | 2.675.560 |
| Registros con capacidad positiva y no ambigua | 2.675.560 |
| Registros sin capacidad válida | 0 |
| Cobertura de registros con capacidad válida | 100,00 % |

La cobertura completa de registros no elimina los problemas de calidad del
catálogo: las cuatro estaciones con capacidad cero no aparecen en los datos
estación-hora del periodo auditado.

## Calidad y distribución de capacidad

No se encontraron IDs nulos ni duplicados. Tampoco hay capacidades nulas. Hay
cuatro capacidades inválidas porque `dpcapacity` es cero para los IDs `99`,
`109`, `116` y `138`; ninguna de esas estaciones tiene registros en el periodo
auditado. La validación estricta rechaza el catálogo completo mientras esas
filas permanezcan, pero el comando de auditoría las conserva y reporta en lugar
de eliminarlas.

Entre las 480 capacidades positivas, el mínimo es 11 y el máximo es 39.

| `dpcapacity` | Estaciones |
|---:|---:|
| 11 | 54 |
| 15 | 201 |
| 19 | 175 |
| 23 | 39 |
| 27 | 2 |
| 30 | 1 |
| 31 | 7 |
| 39 | 1 |

## Limitaciones y uso recomendado

- `dpcapacity` representa la capacidad del catálogo consultado, no el número de
  bicicletas o anclajes disponibles en una hora.
- La versión es posterior a los viajes de 2025. No debe usarse como capacidad
  histórica validada ni como evidencia de inventario instantáneo.
- La cobertura del 100 % describe coincidencia de IDs y capacidad positiva del
  catálogo actual; no valida vigencia histórica.
- Antes de normalizar etiquetas de presión por capacidad, se necesita un
  catálogo de 2025 verificable o evidencia de que las capacidades no cambiaron.
- Esta auditoría no construye etiquetas P2, no entrena clasificadores y no lee
  datos de noviembre o diciembre de 2025.
