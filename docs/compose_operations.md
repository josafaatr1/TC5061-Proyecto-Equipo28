# Operacion local con Docker Compose

## Arquitectura

[docker-compose.yml](../docker-compose.yml) inicia dos servicios en el proyecto
`mibici-compose`, sin nombres fijos de contenedor:

| Servicio | Acceso desde Windows | Configuracion |
|---|---|---|
| `api` | `http://127.0.0.1:8000` | Build desde la raiz con el [Containerfile](../Containerfile) original; usuario `app`, UID 10001. |
| `mlflow` | `http://127.0.0.1:5000` | Imagen oficial 3.17.0 fijada por tag y digest; SQLite y artefactos persistentes. |

La red bridge `mibici-compose_mibici` resuelve `mlflow` por nombre.
La API recibe `MLFLOW_TRACKING_URI=http://mlflow:5000`, pero **la inferencia no
consulta ese servidor por solicitud**: conserva `BIKE_MODEL_URI` y
`DOCK_MODEL_URI` del Containerfile y carga los modelos empaquetados al iniciar.
El Registry de origen, la exportacion y el servidor Compose son etapas distintas.
Un volumen nuevo de Compose no contiene los modelos registrados del equipo.
No se monta ni se modifica el `mlflow.db` local preexistente.

MLflow usa `sqlite:////mlflow/mlflow.db`, sirve artefactos por HTTP y los guarda
en `/mlflow/artifacts`. Ambos destinos usan `mibici-compose_mlflow-data`.
Un solo worker evita multiplicar procesos para esta demostracion local con SQLite;
no constituye una optimizacion de capacidad medida.

MLflow se comprueba con `/health` cada 10 s (timeout 5 s, inicio 60 s, 12 reintentos).
`api` espera `service_healthy` y hereda el healthcheck original de `/health`
(30 s, timeout 5 s, inicio 30 s, 3 reintentos).
Esta dependencia ordena el arranque, no supervisa ni reinicia dependientes si MLflow
cae posteriormente. Ambos servicios usan `restart: unless-stopped` y logs
`json-file` limitados a tres archivos de 10 MB por contenedor.

Los puertos se publican solo en loopback. Ambos contenedores eliminan todas las
capabilities y usan `no-new-privileges`; MLflow conserva el UID 0 de la imagen
oficial, comprobado con acceso de escritura al volumen. No es un despliegue de
produccion: no incluye autenticacion, TLS, aislamiento entre usuarios del host
ni una certificacion de seguridad. El middleware de MLflow sigue habilitado;
solo admite hosts localhost, 127.0.0.1 y `mlflow:5000`.

## Prerrequisitos y modelos autenticos

Desde la raiz, en PowerShell:

```powershell
docker version
docker compose version
docker info --format '{{.OSType}}'
docker ps -a
Get-NetTCPConnection -State Listen -LocalPort 8000,5000 -ErrorAction SilentlyContinue
```

Se requiere Docker Desktop con motor Linux y Compose con soporte para
`service_healthy`. Si el motor esta detenido, ejecutar `docker desktop start`.
Si hay un puerto ocupado, no detener contenedores ajenos: coordinar con su
propietario antes de iniciar Compose.

El clon de Git no incluye los modelos. Antes del build deben existir estos
archivos y todos los archivos auxiliares de cada exportacion:

```powershell
$required = @(
    '.\models\p2-bike-pressure\MLmodel',
    '.\models\p2-bike-pressure\model.skops',
    '.\models\p2-dock-pressure\MLmodel',
    '.\models\p2-dock-pressure\model.skops',
    '.\models\metadata.json'
)
$missing = $required | Where-Object { -not (Test-Path $_) }
if ($missing) { throw "Faltan artefactos autenticos: $($missing -join ', ')" }
Get-Content .\models\metadata.json
```

Si faltan, obtener del responsable de API/Locust las exportaciones completas y
su metadata, o usar el Registry **existente y de confianza** con ambos aliases
`champion`. Con Python 3.14 y el venv de desarrollo instalado segun el
[README, seccion 14.2](../README.md#142-prerrequisitos-y-exportacion):

```powershell
$env:PYTHONUTF8 = '1'
# Solo si esta URL corresponde al Registry de origen, no a un Compose vacio.
$env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
.\.venv\Scripts\python.exe -m mibici.api.export_models
Get-Content .\models\metadata.json
```

Para un Registry remoto, usar su URL autorizada y configurar credenciales solo
fuera de Git. El exportador reemplaza las carpetas de destino: no ejecutarlo
sobre modelos locales validos sin respaldo y necesidad de actualizacion.
No entrenar, registrar nuevas versiones ni cambiar aliases para cubrir faltantes.
Los modelos, metadata, DB y artefactos locales permanecen excluidos de Git.

## Arranque y operacion

Con Docker y las exportaciones aprovisionadas, el arranque es un comando:

```powershell
docker compose up -d --build
```

La orden vuelve al terminal antes de que necesariamente termine la carga de la
API. Para esperar salud y comprobar estado:

```powershell
docker compose up -d --wait --wait-timeout 180
docker compose ps
docker compose logs --tail 100 api
docker compose logs --tail 100 mlflow
docker compose stats --no-stream
```

Operaciones limitadas a este proyecto:

```powershell
docker compose stop
docker compose start
docker compose restart
docker compose down
docker compose up -d --build
```

`down` elimina contenedores y red, pero conserva el volumen nombrado. Mantener
el nombre del proyecto y del volumen al volver a iniciar; cambiar `-p` o
`COMPOSE_PROJECT_NAME` selecciona otro volumen. **No usar `down -v` ni borrar
`mibici-compose_mlflow-data`**, pues destruye DB y artefactos. La persistencia
no sustituye un respaldo.

## Pruebas reproducibles

```powershell
docker compose config --quiet
docker compose config --services
$env:PYTHONUTF8 = '1'
Invoke-RestMethod -Uri http://127.0.0.1:8000/health
$payload = Get-Content .\docs\examples\p2-request.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType 'application/json' -Body $payload
(Invoke-WebRequest -Uri http://127.0.0.1:5000/ -UseBasicParsing).StatusCode
docker compose exec -T api python -c 'import urllib.request; print(urllib.request.urlopen("http://mlflow:5000/health", timeout=5).status)'
$env:MIBICI_API_CONTAINER = '1'
$env:MIBICI_API_URL = 'http://127.0.0.1:8000'
.\.venv\Scripts\python.exe -m pytest tests\api\test_main.py -q
Remove-Item Env:MIBICI_API_CONTAINER, Env:MIBICI_API_URL
.\.venv\Scripts\python.exe -m ruff format --check
.\.venv\Scripts\python.exe -m ruff check
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
.\.venv\Scripts\python.exe -m pytest -q
```

Los limites de hilos afectan solo los procesos lanzados desde esa terminal y
evitan sobreparalelismo durante los tests numericos en Windows. No modifican
los servicios Compose ni el entrenamiento del proyecto.

Exigir HTTP 200, `models_loaded` con bike/dock, ambas predicciones binarias y
probabilidades entre 0 y 1. La prueba HTTP optativa comprueba tambien OpenAPI
y tres entradas invalidas con 422. Los unitarios usan mocks; no son evidencia
de inferencia real. No habilitar la integracion de Registry contra un volumen
Compose nuevo: ese Registry no contiene los champions de origen.

### Persistencia de DB y artefactos

Crear solo un experimento propio, guardar su ID, subir un artefacto y recuperarlo
despues de recrear contenedores. No modificar experimentos del equipo:

```powershell
$base = 'http://127.0.0.1:5000'
$name = "compose-persistence-$([guid]::NewGuid().ToString('N'))"
$created = Invoke-RestMethod -Method Post -Uri "$base/api/2.0/mlflow/experiments/create" -ContentType 'application/json' -Body (@{name=$name} | ConvertTo-Json)
$id = $created.experiment_id
$createdRun = Invoke-RestMethod -Method Post -Uri "$base/api/2.0/mlflow/runs/create" -ContentType 'application/json' -Body (@{experiment_id=$id;run_name='compose-volume-check';start_time=[DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds()} | ConvertTo-Json)
$runId = $createdRun.run.info.run_id
$root = $createdRun.run.info.artifact_uri
if (-not $root.StartsWith('mlflow-artifacts:/')) { throw "URI de artefactos no esperada: $root" }
$artifactPath = $root.Substring('mlflow-artifacts:/'.Length) + '/checks/persistence.txt'
$artifactUrl = "$base/api/2.0/mlflow-artifacts/artifacts/$artifactPath"
$content = "MiBici Compose persistence check`n"
Invoke-WebRequest -Method Put -Uri $artifactUrl -ContentType 'text/plain' -Body $content -UseBasicParsing
Invoke-RestMethod -Method Post -Uri "$base/api/2.0/mlflow/runs/update" -ContentType 'application/json' -Body (@{run_id=$runId;status='FINISHED'} | ConvertTo-Json)
"Experimento=$id Run=$runId Nombre=$name"
docker compose down
docker compose up -d --build --wait --wait-timeout 180
$recovered = Invoke-RestMethod -Uri "$base/api/2.0/mlflow/experiments/get?experiment_id=$id"
if ($recovered.experiment.name -ne $name) { throw 'No se recupero el experimento.' }
$downloaded = Invoke-WebRequest -Uri $artifactUrl -UseBasicParsing
if ($downloaded.StatusCode -ne 200 -or $downloaded.Content -ne $content) { throw 'El artefacto no persistio.' }
```

## Diagnostico y compatibilidad con Locust

| Sintoma | Siguiente comprobacion |
|---|---|
| API no inicia | Logs de `api`; comprobar exportacion completa y versiones sklearn/skops del MLmodel. Reconstruir despues de aprovisionar. |
| MLflow no saludable | Logs de `mlflow`; verificar SQLite, montaje `/mlflow` y permisos. No borrar DB para ocultar un error. |
| HTTP 403 en MLflow | Revisar Host usado; la red admite `http://mlflow:5000`, no nombres arbitrarios. No deshabilitar el middleware. |
| Puerto ocupado | `docker ps -a` y `Get-NetTCPConnection`; coordinar el servicio propio sin detener uno ajeno. |
| UnicodeEncodeError en cliente MLflow | Definir `PYTHONUTF8=1` antes de ejecutar Python con salida redirigida. |

[Locust](../tests/load/README.md) documenta el contenedor independiente
`mibici-api-baseline`. No puede compartir el puerto 8000 con Compose. Locust
puede usar el mismo `--host http://127.0.0.1:8000`; para monitoreo, obtener el
contenedor actual con `$apiContainer = docker compose ps -q api` y pasarlo a
`.\.venv\Scripts\python.exe .\tests\load\monitor_resources.py --container $apiContainer --output <archivo-nuevo.csv>`.
Sustituir ese marcador por un archivo nuevo: no sobrescribir resultados
historicos. No se cambiaron scripts ni se repitieron escenarios de carga.

Resultados de esta ejecucion: [evidencia Compose](evidence/compose/validation.md).
Mejoras futuras: respaldo/restauracion probados, limites de recursos sustentados
por mediciones y una imagen MLflow sin root si se requiere despliegue fuera
de la demostracion local. No se agregaron Redis, colas ni PostgreSQL.
