# Revisión básica de seguridad

Fecha: 2026-11-01 · Rama: `feature/security-checks`

## 1. Credenciales y datos sensibles

| Revisión | Resultado |
|---|---|
| Archivos versionados (`git ls-files`) con nombres tipo `.env`, llaves, certificados, bases de datos | Ninguno. Solo código, documentación, manifiestos de fuentes y `.gitkeep`. |
| Búsqueda de `password`, `secret`, `token`, `api_key`, `aws_`, URLs con usuario | Sin coincidencias en el código. |
| Historial completo de Git (`git log --all -p`) | Sin credenciales en commits anteriores. |
| `detect-secrets` sobre los archivos versionados | 4 alertas en `notebooks/01_exploracion_mibici_2025.ipynb`; todas son imágenes PNG en base64 (falsos positivos). Se excluyen con `--exclude-lines '"image/png":'`. |
| Datos y modelos | `data/**`, `models/*`, `mlruns/`, `mlartifacts/` y `mlflow.db` ya estaban en `.gitignore`. |

Cambios:

- `.gitignore`: se añaden llaves y certificados (`*.pem`, `*.key`, `id_rsa*`…), archivos de credenciales (`credentials*.json`, `service-account*.json`, `secrets/`), entornos `.venv*/` y `reports/security/`. `.env` y `.env.*` ya estaban ignorados.
- `.env.example`: plantilla de variables de entorno (`MLFLOW_EXPERIMENT_NAME`, `MLFLOW_TRACKING_URI` y, comentadas, las credenciales de un servidor remoto). `mibici/config.py` ya carga `.env` con `python-dotenv`, así que la configuración se toma de variables de entorno y no del código.

## 2. Dependencias (`pip-audit`)

`pip-audit -r requirements-lock.txt` encontró **50 vulnerabilidades conocidas en 6 paquetes**, todas en dependencias transitivas y con versión corregida:

| Paquete | Versión anterior | Vulnerabilidades | Versión corregida |
|---|---|---|---|
| pillow | 12.3.0 | 39 | 13.0.0 |
| pyasn1 | 0.6.4 | 4 | 0.6.5 |
| tornado | 6.5.10 | 3 | 6.6 |
| certifi | 2026.7.22 | 2 | 2026.10.18 |
| mlflow | 3.17.0 | 1 | 3.17.2 |
| GitPython | 3.2.0 | 1 | 3.2.1 |

Se actualizaron `requirements-lock.txt`, `requirements.txt` y `python_env.yaml` (mlflow, mlflow-skinny y mlflow-tracing a 3.17.2). Tras el cambio, `pip-audit` reporta **0 vulnerabilidades**. El paquete local `mibici` se omite porque no está en PyPI.

## 3. Análisis estático del código (`bandit`)

`bandit -r mibici` (4 072 líneas) reportó 3 hallazgos de severidad **baja**, todos en `git_commit()` de `mibici/modeling/mlflow_runs.py`:

- B404: importación de `subprocess`.
- B603 / B607: `subprocess.run(["git", "rev-parse", "HEAD"])`.

El comando es fijo, no usa shell y no recibe entrada del usuario, por lo que no hay riesgo de inyección. Se marcaron con `# nosec` y una nota. Resultado final: **0 hallazgos**.

## 4. Cómo repetir la revisión

```bash
python -m pip install -r requirements-security.txt
make security
```

`make security` ejecuta `detect-secrets`, `pip-audit` y `bandit`, y falla si alguna encuentra problemas.
