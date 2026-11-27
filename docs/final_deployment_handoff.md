# Handoff de despliegue: integrante 3

**Cierre técnico local para revisión.** Rama `feature/anthony-final-deployment-operations`, base `4c576efc86d55c828e7101d47628f78ea23b89ff`. La publicación y el CI del nuevo SHA requieren aprobación. No se hizo commit, push, merge ni PR.

## Matriz de aceptación

| Entregable | Estado | Evidencia o pendiente |
|---|---|---|
| Auditoría de rama y CI base | PASÓ | [Run de `main`](https://github.com/josafaatr1/TC5061-Proyecto-Equipo28/actions/runs/45937362101), SHA `4c576ef`. |
| Identidad de los artefactos operativos | PASÓ | Model Card, metadata, `MLmodel`, estimadores y SQLite coinciden. [Manifiesto SHA256](evidence/final_deployment/model_card_export_sha256.json). |
| FastAPI contenerizada y paridad | PASÓ local | 17 hashes en host e imagen; HTTP real, OpenAPI, errores 422 y paridad para 3 casos × 2 modelos. 153 pruebas API aprobadas. |
| Docker Compose integrado | PASÓ local | API y MLflow `healthy` en el proyecto aislado; `docker compose config` y build pasaron. |
| Pruebas de carga Locust | PASÓ local | 1, 5, 10 y 20 usuarios; 1135 solicitudes HTTP 200, 0 fallos. [CSV](evidence/final_deployment/model_card_load/load_summary.csv) y [gráfico](evidence/final_deployment/model_card_load/load_summary.png). |
| Regresión y evidencia reproducible | PASÓ local | 320 pruebas aprobadas, 3 skips optativos; Ruff sin hallazgos. [Comandos](final_deployment_operations.md) y [registro de cierre](evidence/final_deployment/model_card_validation.md). |
| Conciliación documental con el Registry de #34 | Pendiente del equipo | Se conserva la discrepancia; no bloquea la referencia operativa del Model Card indicada para la demo. |
| CI en GitHub del SHA final | NO EJECUTADO | Revisión y aprobación antes de publicar; verificar `headSha` y los pasos del run real. |
| Capturas definitivas y video | NO EJECUTADO | Capturar el build definitivo en la demo; no hay grabación ni capturas simuladas. |

### Conciliación con el integrante 2

El [informe de operación](final_deployment_operations.md#linaje-operativo-y-discrepancia-del-registry) contrasta runs y AP de ambos registros. Mostrar en la demo los IDs del Model Card. Si el equipo cambia el artefacto o la configuración, repetir preflight, build, Compose, hashes, HTTP/paridad, tests y carga; no reutilizar esta evidencia como si correspondiera al nuevo build.

### Datos para el integrante 4

Compartir SHA, ID/digest de imagen, runs, plataforma, workers, versión del runtime, límites CPU/RAM, número de muestras, ventana de benchmark, requests y errores, RPS y percentiles. El benchmark de usuarios con espera no actualiza capacidad máxima ni costo por inferencia. Si cambian significativamente los artefactos o los límites, decidir con el integrante 4 si repetir `locustfile_capacity.py` con la configuración de FinOps. No sustituir el informe previo por estos resultados.

## Archivos y revisión

| Archivo | Cambio |
|---|---|
| [README](../README.md#16-operación-del-sistema-entregable-7) | Enlace al informe. |
| [Preflight](../mibici/api/preflight.py) y [tests](../tests/api/test_preflight.py) | Linaje offline y SHA256 de todo el export; rechaza ausencia, export anidado, runs divergentes y modificaciones frente a la referencia. |
| [Tests de API](../tests/api/test_main.py) | Swagger por HTTP y paridad optativa con tres inputs, sin cambiar el contrato. |
| [Workflow](../.github/workflows/ci.yml) | `docker compose config --quiet` y build claramente sin artefactos de Registry en checkout. |
| [Runner Locust](../tests/load/run_final.py) | Exporta el JSON final, valida conteos contra la API y genera CSV y gráfico. |
| [Manual final](final_deployment_operations.md), [registro de cierre](evidence/final_deployment/model_card_validation.md) y [evidencia preliminar](evidence/final_deployment/validation.md) | Repetición PowerShell y resultados separados por alcance. |

## Slide y demo grupal (50 segundos)

Slide: **Despliegue P2**. Mostrar `export del Model Card → hashes → Docker Compose → HTTP → Locust`. A pie: SHA, digest y alcance. Para la tabla de carga, usar la ejecución del mismo build, no números preliminares. Si Actions del SHA aprobado todavía no existe, mostrar `main` rotulado como referencia y decirlo en voz alta.

| Tiempo | Mostrar | Decir |
|---|---|---|
| 0–10 s | `docker compose ps` con `api` y `mlflow` en `healthy` | «FastAPI carga dos modelos P2 desde la imagen. El MLflow de Compose hace tracking local.» |
| 10–28 s | Swagger `POST /predict` o PowerShell; HTTP 200 | «Enviamos el ejemplo sintético. La API devuelve clase y probabilidad para bicicletas y anclajes.» |
| 28–39 s | Actions, con el SHA visible | «Este run de `main` pasó pruebas, Ruff y build. El cambio de Compose se comprobó localmente. Ningún paso publica modelos ni despliega a producción.» |
| 
| 39–50 s | Tabla Locust del mismo build: usuarios, errores, P95 | «Medimos cuatro niveles con espera entre peticiones; esto no mide capacidad máxima.» |

Capturas manuales: `compose ps` y fecha; Swagger con entrada de ejemplo, HTTP 200 y ambos resultados; Actions con SHA y pasos visibles; Locust con escenario, duración, requests, errores y percentiles. Referenciar la evidencia del Registry del integrante 2 en vez de copiarla. Si algún requisito sigue bloqueado, enseñar únicamente el resultado realmente obtenido y nombrar lo que falta.

### Comandos para capturas y demo

Desde la raíz, con los archivos y el manifiesto [verificados](final_deployment_operations.md#1-verificar-origen-y-exportar-sin-sobrescribir), usar un proyecto propio. No levantarlo si 18000 o 15000 ya están ocupados.

```powershell
Get-NetTCPConnection -State Listen -LocalPort 18000,15000 -ErrorAction SilentlyContinue
$override = Join-Path $env:TEMP ("mibici-demo-" + [guid]::NewGuid().ToString('N') + ".yml")
@'
services:
  api:
    image: mibici-api:int3-modelcard-demo
    ports: !override
      - "127.0.0.1:18000:8000"
  mlflow:
    ports: !override
      - "127.0.0.1:15000:5000"
'@ | Set-Content -LiteralPath $override -Encoding utf8
$c = @('compose','-p','mibici-int3-demo','-f','docker-compose.yml','-f',$override)
docker @c config --quiet
if ($LASTEXITCODE -ne 0) { throw 'Configuración Compose inválida.' }
docker @c up -d --build --wait --wait-timeout 300
if ($LASTEXITCODE -ne 0) { throw 'No iniciar la demo: Compose no está healthy.' }
Get-Date -Format o
docker @c ps
docker inspect --format 'image={{.Image}} health={{.State.Health.Status}}' (docker @c ps -q api)
```

Registrar el digest de la nueva instancia y cotejar el contenido. Si no es el digest medido, identificar la carga como evidencia de la imagen original y repetir Locust para hacer una afirmación sobre el nuevo build.

```powershell
Get-Content -Raw $reference | docker @c exec -T api python -m mibici.api.preflight `
    --models-dir /app/models @lineage --expected-report /dev/stdin
if ($LASTEXITCODE -ne 0) { throw 'El contenido de la imagen no coincide.' }
$apiUrl = 'http://127.0.0.1:18000'
Invoke-RestMethod "$apiUrl/health"
$body = Get-Content -Raw .\docs\examples\p2-request.json
Invoke-WebRequest "$apiUrl/predict" -Method Post -ContentType application/json -Body $body |
    Select-Object StatusCode,Content
Start-Process "$apiUrl/docs"
```

En Swagger: `POST /predict` → `Try it out` → pegar `p2-request.json` → `Execute`. Capturar `200` y el cuerpo con ambos objetivos. Para Actions, abrir el run real por SHA; mientras el SHA final no esté publicado, usar el enlace de `main` rotulado como referencia. Para Locust, ejecutar los comandos de abajo. El resumen pertenece al digest de la imagen medido, no a cualquier build de la demo. No hace falta repetir la prueba en vivo.

```powershell
Start-Process 'https://github.com/josafaatr1/TC5061-Proyecto-Equipo28/actions/runs/45937362101'
Import-Csv .\docs\evidence\final_deployment\model_card_load\load_summary.csv |
    Format-Table users,requests,failures,requests_per_second,mean_ms,p95_ms,p99_ms
Invoke-Item .\docs\evidence\final_deployment\model_card_load\load_summary.png
```

Al terminar, `docker @c down` sin `-v`.

## Publicación (solo después de aprobación)

Preparar exclusivamente los archivos de la tabla anterior y el directorio de evidencia nuevo. No usar `git add .`: hay archivos locales no rastreados de otros trabajos. Verificar primero `git diff --cached --stat` y `git diff --cached --check`. Comandos sugeridos, no ejecutados:

```powershell
git add -- .github\workflows\ci.yml README.md mibici\api\preflight.py `
    tests\api\test_main.py tests\api\test_preflight.py tests\load\run_final.py docs\final_deployment_operations.md `
    docs\final_deployment_handoff.md docs\evidence\final_deployment
git diff --cached --stat
git diff --cached --check
git commit -m "docs(ops): validate P2 deployment and document final model handoff" `
    -m "Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
git push -u origin feature/anthony-final-deployment-operations
gh pr create --base main --head feature/anthony-final-deployment-operations
```

`gh` no estaba instalado durante la auditoría. Usar la interfaz de GitHub si no se dispone de la CLI. No declarar CI en verde hasta que el run del SHA publicado termine correctamente.
