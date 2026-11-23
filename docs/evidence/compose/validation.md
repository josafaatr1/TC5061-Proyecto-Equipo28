# Evidencia de Docker Compose

Ejecucion real del 22/11/2026 en Windows PowerShell y Docker Desktop.
Los logs Docker usan UTC (23/11/2026); Lima corresponde a UTC-5.
No se reutilizan resultados historicos como pruebas de esta entrega.

## Base y alcance

- Rama `feature/anthony-docker-compose`, inicialmente igual a `origin/main`:
  `a7e62ab931415d737bc857a2612ad20214a01e41`.
- PR #29 y #30 confirmados como integrados mediante Git y API GitHub.
- Docker Desktop se inicio con `docker desktop start`; motor Linux 29.8.0,
  cliente 29.8.0 y Compose v5.5.1.
- Venv del proyecto: Python 3.14.4. El Python global 3.13.7 no se uso para tests.
- Puertos 8000 y 5000 libres antes de Compose. Los dos contenedores preexistentes
  de otro proyecto permanecieron detenidos y sin cambios.
- ZIP y reportes locales ajenos no se incorporaron a esta contribucion.
- No se alteraron API, entrenamiento, modelos, aliases, dependencias, CI ni Locust.

## Imagen y modelos

Se ejecutaron `docker manifest inspect`, `docker pull`, `docker image inspect`
y pruebas de Python/`mlflow server --help` sobre la imagen oficial:

```text
ghcr.io/mlflow/mlflow:v3.17.0
Digest: sha256:78803b7afd156eced665b023554bc199a4723d4466e7cba3732f7284f3cddd11
Plataformas verificadas: linux/amd64, linux/arm64
Python de la imagen MLflow: 3.11.15
MLflow: 3.17.0
Entrypoint: null; Cmd: ["python3"]; UID por defecto: 0
```

El servicio define explicitamente `mlflow server`. No se supuso que la imagen
arrancara un servidor ni que incluyera curl. Su Python sirve el healthcheck.
El volumen `/mlflow` fue comprobado escribible; SQLite se creo alli.

Los modelos locales ya existian y no se reexportaron ni reemplazaron.
Metadata: exportacion `2026-11-09T02:24:54+00:00`, Registry de origen
`http://127.0.0.1:5000`, ambos aliases `champion`, version 1:

| Modelo | Run original |
|---|---|
| `mibici-p2-bike-pressure` | `d306fe5f7f34496a94f8cc3d0db2f62b` |
| `mibici-p2-dock-pressure` | `81f93c6f160e4f44afe3db913eac5030` |

Ambos MLmodel declaran MLflow 3.17.0, sklearn 1.9.1, formato skops y firma
`float32 [-1, 15]`. Se cargaron los artefactos reales en el host y se compararon
sus predicciones/probabilidades con el HTTP del contenedor: igualdad exacta para
la entrada de ejemplo. No se crearon modelos ni se usaron mocks en esta prueba.

## Comandos y resultados

| Comprobacion ejecutada | Resultado |
|---|---|
| `docker compose config --quiet` | Exit 0. |
| `docker compose config --services` | Exactamente `mlflow` y `api`. |
| `docker compose up -d --build` | Exit 0, imagen API construida con Containerfile original. |
| `docker compose up -d --wait --wait-timeout 180` | Exit 0, ambos servicios healthy. |
| `docker compose logs --tail 100 api` / `mlflow` | Arranque completo; 200 en healthchecks; sin fallo de carga. |
| API `GET /health` | HTTP 200, `{"status":"healthy","models_loaded":["bike","dock"]}`. |
| API `POST /predict` | HTTP 200, respuesta real debajo. |
| MLflow `GET /` | HTTP 200. |
| Desde `api`, `http://mlflow:5000/health` | HTTP 200; DNS y comunicacion por red Compose probados. |
| Con `mlflow` detenido, API `/health` y `/predict` | HTTP 200 en ambos; inferencia local sin servidor MLflow. Se restauro MLflow y ambos quedaron healthy. |
| Inspeccion de seguridad efectiva | API UID 10001; ambos cap_drop ALL y no-new-privileges; puertos solo 127.0.0.1; API sin montajes. |
| `git check-ignore` sobre modelos y DB | Todos excluidos de Git. |
| Pytest API sin flags | 143 passed, 2 skipped, 1 warning; 43.86 s. |
| Pytest API con `MIBICI_API_CONTAINER=1` | 144 passed, 1 skipped, 1 warning; 21.06 s. |
| `python -m ruff format --check` | Exit 0, 25 files already formatted. |
| `python -m ruff check` | Exit 0, All checks passed. |
| `python -m pytest -q` con hilos numericos limitados a 1 | Exit 0: 294 passed, 2 skipped, 117 warnings; 1446.27 s (24:06). |

El test HTTP real tambien comprobo OpenAPI y respuestas 422 para hora 24,
campo desconocido y NaN. La integracion de Registry se omitio intencionalmente:
el nuevo volumen Compose no contiene los champions de origen. Los unitarios
existentes usan mocks, separados de los HTTP reales.

La primera suite completa se interrumpio despues de mas de 30 minutos por
lentitud, sin resumen final ni aprobacion declarada. Se ejecuto el caso existente
`test_evaluate_candidate_reuses_trained_model` con OMP/OPENBLAS/MKL_NUM_THREADS=1:
1 passed, 12 deselected, 9.04 s de Pytest (28.08 s de proceso).
Se relanzo la suite completa con esos limites y `PYTHONUTF8=1`, solo en el
proceso de validacion. No se cambiaron tests, codigo, dependencias ni ajustes
globales para acelerar o eludir comprobaciones.
La segunda ejecucion termino correctamente. Las 117 advertencias incluyen
deprecaciones de Starlette, Evidently/NumPy, MLflow, SQLAlchemy y Deepchecks;
firmas con enteros, fuentes de datos ambiguas, puntos sinteticos duplicados
en clustering y division por cero en una comparacion de baseline sintetico.
No se ocultaron ni se corrigieron asuntos preexistentes fuera de este alcance.

Respuesta obtenida con [p2-request.json](../../examples/p2-request.json), entrada
sintetica de prueba identificada por la documentacion original:

```json
{"predictions":{"bike":{"prediction":0,"probability":0.3171913481120226},"dock":{"prediction":1,"probability":0.5846188145068164}},"decision_threshold":0.5}
```

`docker compose stats --no-stream` se ejecuto sin carga Locust: API 0.33% CPU,
188.4 MiB RAM; MLflow 2.22% CPU, 1.611 GiB RAM. Es una captura puntual del
entorno, no promedio, benchmark, costo monetario ni evidencia de optimizacion.

## Persistencia real

1. Se creo exclusivamente el experimento propio
   `compose-persistence-20261123T002644Z-eacb1749`, ID **1**.
   Run `91c8901be6de48e9b6e26f0aa926c66e`; artefacto
   `checks/persistence.txt`, contenido `MiBici Compose persistence check`
   seguido de CRLF.
2. Se subio y descargo por cliente MLflow, luego se marco el run `FINISHED`
   mediante REST.
3. `docker compose restart` y espera de salud: experimento recuperado.
4. `docker compose down`: contenedores y red eliminados; `docker volume inspect
   mibici-compose_mlflow-data` confirmo volumen conservado.
5. `docker compose up -d --build` y espera de salud: nuevos IDs de contenedor,
   experimento recuperado por REST con HTTP 200; run `FINISHED` y artefacto
   descargado con HTTP 200. Descarga por cliente tambien aprobada.
6. Se ejecuto la receta PowerShell de la [guia](../../compose_operations.md)
   creando el experimento propio **2**,
   `compose-persistence-3d6c142cd13f48ae917038fe8c119273`,
   run `d940ed487d2a4d08a9a04172c9cb71ea`, con contenido LF.
   Otro `down/up --build --wait` recupero experimento y contenido exacto con
   HTTP 200. El artefacto del experimento 1 tambien siguio intacto.
7. Despues de recrear, API healthy y `/predict` HTTP 200 con ambos resultados.

SHA256 del artefacto original recuperado:
`d3335a12f498ff621872eb5384901a454a6b20f221e4d986978cfcebd8eecdfb`.
La comparacion final verifico el contenido exacto, incluidos finales de linea.
No se ejecuto `down -v`, ni se borraron experimentos o volumenes.

Dos incidencias de verificacion se resolvieron sin cambiar el servicio:
el cliente MLflow intento imprimir un emoji con cp1252 y fallo al terminar el
run; se uso salida UTF-8 y REST para finalizarlo. Una comparacion inicial
esperaba LF, pero `mlflow.log_text` en Windows produjo CRLF; se inspeccionaron
los bytes y se verifico el contenido real. No fueron perdidas de persistencia.

Ultimo estado local comprobado:

```text
SERVICE   CONTAINER                  STATUS     PORT
api       mibici-compose-api-1       healthy    127.0.0.1:8000->8000/tcp
mlflow    mibici-compose-mlflow-1    healthy    127.0.0.1:5000->5000/tcp
VOLUME    mibici-compose_mlflow-data (conservado)
```

## Limitaciones y control de contenido

La guia distingue aprovisionamiento previo de un arranque con un comando.
Un clon limpio sin exportaciones autenticas no puede servir inferencia.
La disponibilidad del Registry del equipo no se ha vuelto a certificar:
se reutilizaron modelos exportados validos, sin modificarlos.
No se repitio Locust ni FinOps; su compatibilidad se documento sin sobrescribir
scripts/resultados. MLflow conserva root de la imagen oficial; se limita a
loopback y a capacidades eliminadas, sin autenticacion/TLS ni respaldo probado.

Antislop se aplico durante la redaccion, por eleccion de esta sesion.
Alcance backend/documentacion, sin UI nueva ni certificacion visual de MLflow.

- R-02 PASS: el texto nuevo no usa rayas largas.
- R-17/C-5 PASS: versiones, conteos, memoria, HTTP e IDs proceden de ejecuciones observadas.
- R-35 PASS: build, salud, prediccion valida, entradas invalidas y persistencia ejecutados.
- R-36/R-38 PASS: limitaciones explicitas; entrada sintetica identificada, modelos y respuestas reales.

Publicacion y GitHub Actions se informaran con la URL real del PR y el SHA
publicado; no se declara CI verde antes de consultar su resultado.
