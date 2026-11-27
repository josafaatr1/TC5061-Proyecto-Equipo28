# Registro de ejecución: despliegue P2

Fecha: 25/11/2026, Lima (UTC−5). Rama `feature/anthony-final-deployment-operations`, HEAD y `origin/main`: `a434cbdbc7536b908092bc2b5b546325c41656c7`. Los resultados locales incluyen cambios de worktree sin commit. **Preliminar:** los modelos usados son los históricos del export local, no los declarados como definitivos en #34.

## Auditoría

| Verificación | Resultado |
|---|---|
| `git status --short --branch`, `git remote -v`, `git fetch origin --prune`, `git log -1 --oneline HEAD`, `git log -1 --oneline origin/main`, `git diff --stat origin/main...HEAD` | Fetch exit 0; base idéntica, diff vacío antes de trabajar. La rama de despliegue no aparece en `git ls-remote --heads origin`. No se hizo merge. Se conservaron los archivos no rastreados preexistentes. |
| PR #34 | Consultado en GitHub MCP porque `gh` no estaba instalado. Abierto, head `dc52a5d1fd02f00d526933940de8e3b000d1edd7`; `git diff --name-only origin/main...origin/feature/miguel-model-registry` confirma solo el Markdown y el PDF. |
| Actions histórico | [Run 45719392530](https://github.com/josafaatr1/TC5061-Proyecto-Equipo28/actions/runs/45719392530), `push` a `main`, SHA `a434cbdbc7536b908092bc2b5b546325c41656c7`, `success`. El job confirmó instalación, pytest, formato, lint y build. No valida el worktree ni arranque con modelos reales. |
| `docker compose config --quiet` / `config --services` | Exit 0; `mlflow` y `api`. |
| Espacio y puertos | Unidad C: 43 GiB libres al auditar. Los contenedores `mibici-compose-api-1` y `mibici-compose-mlflow-1` estaban saludables y escuchando en 127.0.0.1:8000 y :5000, respectivamente. Se dejaron intactos. |
| Excluidos | `git ls-files models` devolvió `models/.gitkeep`; `git check-ignore` confirmó metadata, MLmodel y pesos. Datos, venv y DB preexistentes no se empaquetaron ni se añadieron a Git. |
| Registry accesible | SQLite local en modo `ro`: los dos `champion` v1 tienen los mismos runs históricos del export y los runs de #34 no aparecen en la tabla `runs`. GET de aliases en el **otro** MLflow, el de Compose, devolvió 404 para ambos. `MLFLOW_TRACKING_URI` no estaba definida y no había URL autorizada para el Registry definitivo. |

Host Intel Core Ultra 7 155U, 12 núcleos físicos y 14 lógicos. Docker Desktop `desktop-linux`, engine/cliente 29.8.0, Compose v5.5.1; VM Linux: 14 CPU y 33 372 114 944 bytes de RAM. El venv existente usa Python 3.14.4, MLflow 3.17.0, sklearn 1.9.1, skops 0.16.0, NumPy 2.5.4, pandas 3.0.6, pytest 9.1.1 y Ruff 0.17.0. FastAPI 0.143.0 y Uvicorn 0.54.0 del host no coinciden con los pins de la imagen (0.121.3 y 0.38.0); no se alteró el venv. Los tests HTTP apuntan a la imagen; los unitarios usan el entorno del host. Locust no estaba instalado en ese venv.

## Linaje preliminar

| Modelo | Alias | Versión | Run del export |
|---|---|---:|---|
| `mibici-p2-bike-pressure` | `champion` | 1 | `d306fe5f7f34496a94f8cc3d0db2f62b` |
| `mibici-p2-dock-pressure` | `champion` | 1 | `81f93c6f160e4f44afe3db913eac5030` |

Ambos `MLmodel` declaran los mismos runs que la metadata, formato sklearn/skops, entrada `float32 [-1,15]` y salida `int8 [-1]`. Los requisitos exportados de MLflow, sklearn, skops, NumPy, pandas y SciPy coinciden con las versiones fijadas para la imagen (`mlflow-skinny` en lugar del servidor completo). El preflight con los IDs de #34 devolvió exit 1 en el primer modelo; la revisión manual de metadata/SQLite identificó la discrepancia de ambos. Con los dos IDs históricos devolvió exit 0. `export_models --help` confirmó `--alias` y `--output-dir`; no se ejecutó el export contra un origen no confirmado.

| Ruta relativa a `models/` | SHA256 observado en host |
|---|---|
| `metadata.json` | `e690ff7b5ac880e15de8b5801c254447d8d24aa9dfbe142952a28998b8b9f04d` |
| `p2-bike-pressure/MLmodel` | `f0fbf7680a26c032a5227bba909d43b4ee6c699baa45b0bac1330ab9006257e6` |
| `p2-bike-pressure/model.skops` | `75a58c9c561df13b6a36234ed52acd47f5ec3b993b28e8f98d259e0641b93674` |
| `p2-dock-pressure/MLmodel` | `bb6eba1602c01f6a3bfbcb930cdcbe4c583f89241b43be8e7184dd88bfdd46c9` |
| `p2-dock-pressure/model.skops` | `55a5dfea750767dbdaf4e70240430daf9fa014c0ac5ac24ad7b206bb98fceacd` |

## Arranque preliminar

Override temporal para no tocar el proyecto ni la etiqueta de imagen existentes (`!override` reemplaza los bindings, no agrega puertos):

```yaml
services:
  api:
    image: mibici-api:int3-prelim-20261125
    ports: !override
      - "127.0.0.1:18000:8000"
  mlflow:
    ports: !override
      - "127.0.0.1:15000:5000"
```

Guardar como `$override` fuera del repositorio. Desde la raíz:

```powershell
$c = @('compose','-p','mibici-int3-prelim-20261125','-f','docker-compose.yml','-f',$override)
docker @c config --quiet
docker @c config --services
docker @c up -d --build --wait --wait-timeout 180
docker @c ps
```

Para reproducir HTTP, hashes y carga, usar `docker @c`, la URL `http://127.0.0.1:18000` y los [pasos del informe](../../final_deployment_operations.md#2-construir-probar-por-http-y-cotejar-el-contenido-empacado). Al cerrar, `docker @c down` conserva el volumen de esa prueba. No ejecutar `down -v`.

Docker Compose devolvió exit 0 en `up -d --build --wait --wait-timeout 180` (215.1 s de reloj para el comando completo, incluyendo build y arranque). La imagen local `mibici-api:int3-prelim-20261125` reportó ID y digest de índice `sha256:7b98b7f59f1e263df2c2f4979cd7c171dbc543ca821de2f43c644defedaeb789`, sin publicar.

| Servicio | Estado comprobado | Límites efectivos | Proceso / almacenamiento |
|---|---|---|---|
| `api` | `healthy`, 127.0.0.1:18000 | 1 vCPU, 512 MiB | 1 worker, `OMP_NUM_THREADS=1`, usuario `app`, sin montajes |
| `mlflow` | `healthy`, 127.0.0.1:15000 | 1 vCPU, 2560 MiB | 1 worker, volumen nombrado en `/mlflow` |

La imagen API ejecutó Python 3.14.10, mlflow-skinny 3.17.0, sklearn 1.9.1, skops 0.16.0, FastAPI 0.121.3, Uvicorn 0.38.0, pandas 3.0.6, NumPy 2.5.4 y SciPy 1.18.1. El preflight de `/app/models` coincidió con SHA256 del host en 11 archivos: metadata, ambos MLmodel, ambos pesos y los seis archivos de entorno/requisitos.

## HTTP real preliminar

A las 20:30 de Lima, `Invoke-RestMethod` en `/health` devolvió `{"status":"healthy","models_loaded":["bike","dock"]}`. `/docs`, `/openapi.json` y `/health` de MLflow devolvieron HTTP 200. La API también recibió 200 de `http://mlflow:5000/health` en la red interna.

`POST /predict` con [el ejemplo sintético](../../examples/p2-request.json), HTTP 200:

```json
{"predictions":{"bike":{"prediction":0,"probability":0.3171913481120226},"dock":{"prediction":1,"probability":0.5846188145068164}},"decision_threshold":0.5}
```

Extracto de `docker compose logs --tail 12 api`, sin direcciones de cliente:

```text
INFO:     Started server process [1]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     "GET /health HTTP/1.1" 200 OK
INFO:     "POST /predict HTTP/1.1" 200 OK
INFO:     "GET /docs HTTP/1.1" 200 OK
INFO:     "GET /openapi.json HTTP/1.1" 200 OK
```

MLflow registró `Application startup complete` y `GET /health` 200; su imagen mostró un aviso de deprecación de Starlette, sin impedir el arranque. La persistencia de DB y artefactos sigue respaldada por la [prueba aislada histórica](../compose/validation.md#persistencia-real). En esta ejecución solo se inspeccionó el montaje, sin escribir experimentos.

## Pruebas y lint

| Comando | Resultado |
|---|---|
| `python -m pytest tests\api -q -rs`, `MIBICI_API_CONTAINER=1`, `MIBICI_API_URL=http://127.0.0.1:18000`, `MIBICI_API_MODELS_DIR` apuntando a `models` | PASÓ, exit 0. 150 passed, 1 skipped, 1 warning (56.47 s de pytest; 115.33 s de reloj). Se ejecutaron HTTP real con tres entradas 422 (hora fuera de rango, campo extra y NaN) y paridad de clase/probabilidad para 3 entradas × 2 estimadores. El único skip fue el Registry autenticado (`MIBICI_API_INTEGRATION` no activado). Los tests unitarios con mocks no constituyen evidencia HTTP. |
| `python -m pytest -q -rs`, `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1`, `PYTHONUTF8=1`, sin flags de integración | PASÓ, exit 0. 317 passed, 3 skipped, 94 warnings (1321.31 s de pytest; 1420.69 s de reloj). Los tres skips: HTTP contra contenedor, paridad con export y Registry autenticado. Los dos primeros pasaron en la prueba focalizada; el tercero sigue bloqueado. Se conservaron los avisos de deprecación, firmas MLflow y clustering con datos sintéticos. |
| `python -m ruff format --check`; `python -m ruff check` | Ambos exit 0; 27 archivos en el alcance por defecto. |
| `python -m ruff format --check` y `python -m ruff check` sobre los tres `.py` modificados | Ambos exit 0; incluye los tests fuera de `include` de Ruff. Pylance no encontró errores de sintaxis en esos tres archivos. |

## Locust preliminar

Locust 2.47.0 y matplotlib 3.11.2 se instalaron con `requirements-load.txt` en un venv temporal aislado, sin modificar `.venv`. El 25/11 de Lima (26/11 UTC), después de que terminara la regresión, se ejecutó `locustfile.py` contra `http://127.0.0.1:18000`. Fueron cuatro escenarios secuenciales, 60 s configurados, spawn rate igual a usuarios, espera de 1–2 s. El monitor Docker tomó muestras de la API (`cdfec7c24f11`) y se recortaron a la ventana UTC de `stats_history`. El CSV y la tabla indican qué métricas provienen de la salida final y cuáles del CSV periódico.

| Usuarios | Requests finales / CSV | Fallos de requests | RPS del CSV | Media del CSV (ms) | P95 (ms) | P99 (ms) | CPU media/máx. | RAM media/máx. (MiB) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 37 / 37 | 0 (0 %) | 0.637 | 10.464 | 21 | 36 | 0.71 / 1.95 % | 224.60 / 224.9 |
| 5 | 189 / 186 | 0 (0 %) | 3.304 | 15.668 | 30 | 180 | 3.79 / 33.11 % | 224.92 / 225.4 |
| 10 | 385 / 379 | 0 (0 %) | 6.603 | 18.957 | 42 | 260 | 5.77 / 10.79 % | 225.27 / 225.9 |
| 20 | 757 / 757 | 0 (0 %) | 13.183 | 37.295 | 64 | 820 | 11.94 / 19.36 % | 225.74 / 226.7 |

**PASÓ el contrato bajo carga:** 1368 requests según los totales finales, cero fallos o excepciones del runner; los cuatro procesos salieron con código 0. **FALLÓ el cierre del escritor CSV en 5 y 10 usuarios:** stderr registró `ValueError: I/O operation on closed file` en `StatsCSVFileWriter.stats_writer` después del resumen final. El código instalado de Locust arranca el escritor en un greenlet y llama `close_files()` en el cierre. Faltan 3 y 6 requests en los CSV periódicos frente al resumen final; los P95 y P99 coinciden. `*_failures.csv` y `*_exceptions.csv` tienen encabezado y cero filas; eso no incluye el error del escritor. En la repetición, guardar el JSON final además de conservar stderr, verificar los totales y usar el historial solo para alinear el monitor.

El P99 de 820 ms a 20 usuarios (máximo observado 831.37 ms) muestra cola larga pese al P95 de 64 ms. No se atribuye a saturación: las muestras de CPU no muestran agotamiento sostenido y no se hizo trazado por solicitud. Los [resultados históricos](../../../tests/load/results/graphs/benchmark_summary.csv) del 21/11 reportan P95 de 130 ms y P99 de 540 ms a 20 usuarios; no tienen hashes comparables en ese CSV, su muestreo supera 100 % de un núcleo y no certifican equivalencia de límites, plataforma o build. Esta ejecución fue con 1 vCPU, 512 MiB, un worker y OMP=1. No comparar como mejora de rendimiento ni usar para recalcular FinOps.

[CSV sanitizado](preliminary_load.csv). Los CSV crudos y logs quedan fuera de Git. Al terminar se ejecutó `docker @c down` solo para el proyecto aislado (exit 0), sin borrar el volumen. Los IDs de ambos servicios Compose originales quedaron iguales a los del inicio.
