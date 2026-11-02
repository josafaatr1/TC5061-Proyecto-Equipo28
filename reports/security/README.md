# Auditoría básica de seguridad - MiBici 2025

## Objetivo

Verificar que el repositorio no exponga credenciales ni archivos sensibles y realizar un análisis estático de seguridad del código Python.

## 1. Protección de información sensible

Se revisó y fortaleció el archivo `.gitignore` para excluir archivos de variables de entorno, claves privadas, certificados con clave privada, archivos locales de credenciales, datos del proyecto y artefactos de MLflow.

Se ejecutaron búsquedas sobre los archivos rastreados por Git para identificar nombres asociados con credenciales y referencias a posibles secretos en archivos de configuración y código Python.

**Resultado:** No se encontraron coincidencias con los patrones examinados. Estas verificaciones no constituyen una auditoría exhaustiva del historial de Git ni garantizan la ausencia de secretos.

## 2. Análisis estático con Bandit

**Herramienta:** Bandit 1.9.4  
**Entorno:** Python 3.14.3, Linux  
**Código analizado:** `mibici/`  
**Líneas de código analizadas:** 4,550

Comando utilizado:

```bash
python -m bandit -r mibici/ -f txt
```

### Resultados

| Severidad | Hallazgos |
|---|---:|
| Alta | 0 |
| Media | 0 |
| Baja | 3 |

Los hallazgos corresponden a las reglas B404, B603 y B607, asociadas con el uso de `subprocess` en `mibici/modeling/mlflow_runs.py`.

## 3. Evaluación de hallazgos

La función `git_commit()` ejecuta `git rev-parse HEAD` para obtener el identificador del commit utilizado en la trazabilidad de los experimentos de MLflow.

- **B404:** Importación del módulo `subprocess`. Su presencia no representa por sí sola una vulnerabilidad.
- **B603:** Ejecución de un proceso externo con argumentos constantes, sin utilizar `shell=True` ni entradas proporcionadas por usuarios.
- **B607:** Uso de una ruta parcial para ejecutar `git`. Existe un riesgo residual si la variable de entorno `PATH` está comprometida.

Los tres hallazgos se clasifican como riesgos bajos y se conservan documentados. No se modificó la implementación porque no se identificó una vulnerabilidad explotable directamente en el flujo revisado.

## 4. Evidencia y reproducción

El resultado completo del análisis se encuentra en:

`reports/security/bandit_report.txt`

Para repetir la auditoría:

```bash
python -m pip install bandit==1.9.4
python -m bandit -r mibici/ -f txt
```

Bandit puede devolver un código de salida distinto de cero al detectar hallazgos, aunque el análisis se haya ejecutado correctamente.

## 5. Limitaciones

La revisión se limita a los archivos y patrones inspeccionados, así como al análisis estático del directorio `mibici/`. No incluye una auditoría completa del historial de Git, análisis dinámico, pruebas de penetración ni evaluación de vulnerabilidades de dependencias.