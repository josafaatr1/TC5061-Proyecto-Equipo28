# Análisis FinOps: costo por inferencia de la API MiBici

Generado con `python -m mibici.finops`. No editar a mano: los valores salen de las pruebas de capacidad en `tests/load/results/capacity/`, de los precios en `references/aws_pricing_mx-central-1.json` y de los viajes de MiBici de 2025.

## Supuestos

- Precios on-demand de AWS, región Mexico (Central) (`mx-central-1`), consultados el 2026-11-22; 730 horas por mes.
- Capacidad sostenible: escalón con más peticiones/s con p95 <= 300 ms y errores <= 1%.
- Para dimensionar se usa como máximo el 70% de esa capacidad.
- Las pruebas corrieron en Docker Desktop (AMD Ryzen 7 3700U); un vCPU de AWS puede rendir distinto, así que las capacidades son una referencia, no una garantía.
- t3 y t4g son instancias burstable: sin consumir créditos solo sostienen el 20 % de cada vCPU. Con la demanda real (utilización menor al 10 %) eso basta, pero su costo a capacidad es optimista; para carga sostenida conviene c7g.medium.

## Capacidad medida

| Configuración de la API | Capacidad sostenible | Usuarios | p95 | CPU API prom. | Memoria API máx. | Memoria MLflow máx. |
|---|---|---|---|---|---|---|
| 1 vCPU, 1 proceso | 71.4 req/s | 1 | 20 ms | 96 % | 295 MiB | 1800 MiB |
| 1 vCPU, 1 proceso, OMP=1 | 69.9 req/s | 1 | 20 ms | 96 % | 196 MiB | 1800 MiB |
| 2 vCPU, 1 proceso | 66.6 req/s | 1 | 18 ms | 171 % | 201 MiB | 1811 MiB |
| 2 vCPU, 2 procesos | 29.6 req/s | 2 | 78 ms | 175 % | 413 MiB | 1808 MiB |
| 2 vCPU, 2 procesos, OMP=1 | 95.1 req/s | 16 | 270 ms | 134 % | 410 MiB | 1809 MiB |

Con 16 usuarios concurrentes, un solo proceso con 1 vCPU atiende 26.5 req/s (p95 920 ms) y la configuración optimizada 95.1 req/s (p95 270 ms).

## Demanda real de MiBici (2025)

- 371 estaciones con actividad, 4,532,032 viajes; 12,383 viajes/día en promedio.
- Hora pico: 1,779 viajes. Sin viajes en las horas 1:00, 2:00, 3:00, 4:00 (20 h de operación al día).

| Escenario | Inferencias/mes | Pico (req/s) | Descripción |
|---|---|---|---|
| Lote por estación | 225,692 | 6.18 | Una predicción por estación cada hora de operación (371 estaciones, 20 h/día). |
| App de usuarios | 1,129,912 | 1.48 | 3 consultas por viaje; pico de 1779 viajes/hora. |
| App de usuarios x10 | 11,299,123 | 14.82 | Crecimiento de 10 veces sobre el escenario de app. |

## Costo por inferencia

| Opción | USD/hora | USD/mes | Capacidad | USD por millón (a capacidad) | USD por millón (al 70 %) |
|---|---|---|---|---|---|
| Compose actual: API + MLflow en t3.medium | $0.0437 | $31.90 | 71.4 req/s | $0.1700 | $0.2429 |
| API sola optimizada en t3.small | $0.0218 | $15.91 | 95.1 req/s | $0.0637 | $0.0910 |
| API sola optimizada en t4g.small (Graviton) | $0.0176 | $12.85 | 95.1 req/s | $0.0514 | $0.0734 |
| t4g.small apagada en horas sin servicio | $0.0176 | $10.71 | 95.1 req/s | $0.0514 | $0.0734 |
| Fargate 1 vCPU / 2 GB siempre encendido | $0.0518 | $37.84 | 71.4 req/s | $0.2017 | $0.2881 |

### Costo con la demanda real

| Opción | Escenario | Réplicas | USD/mes | USD por millón | Utilización |
|---|---|---|---|---|---|
| Compose actual: API + MLflow en t3.medium | Lote por estación | 1 | $31.90 | $141.35 | 0.1 % |
| Compose actual: API + MLflow en t3.medium | App de usuarios | 1 | $31.90 | $28.23 | 0.6 % |
| Compose actual: API + MLflow en t3.medium | App de usuarios x10 | 1 | $31.90 | $2.82 | 6.0 % |
| API sola optimizada en t3.small | Lote por estación | 1 | $15.91 | $70.51 | 0.1 % |
| API sola optimizada en t3.small | App de usuarios | 1 | $15.91 | $14.08 | 0.5 % |
| API sola optimizada en t3.small | App de usuarios x10 | 1 | $15.91 | $1.41 | 4.5 % |
| API sola optimizada en t4g.small (Graviton) | Lote por estación | 1 | $12.85 | $56.93 | 0.1 % |
| API sola optimizada en t4g.small (Graviton) | App de usuarios | 1 | $12.85 | $11.37 | 0.5 % |
| API sola optimizada en t4g.small (Graviton) | App de usuarios x10 | 1 | $12.85 | $1.14 | 4.5 % |
| t4g.small apagada en horas sin servicio | Lote por estación | 1 | $10.71 | $47.44 | 0.1 % |
| t4g.small apagada en horas sin servicio | App de usuarios | 1 | $10.71 | $9.48 | 0.5 % |
| t4g.small apagada en horas sin servicio | App de usuarios x10 | 1 | $10.71 | $0.9476 | 5.4 % |
| Fargate 1 vCPU / 2 GB siempre encendido | Lote por estación | 1 | $37.84 | $167.67 | 0.1 % |
| Fargate 1 vCPU / 2 GB siempre encendido | App de usuarios | 1 | $37.84 | $33.49 | 0.6 % |
| Fargate 1 vCPU / 2 GB siempre encendido | App de usuarios x10 | 1 | $37.84 | $3.35 | 6.0 % |

Hallazgo principal: con la demanda de la app, la opción más barata (t4g.small apagada en horas sin servicio) usa el 0.5 % de su capacidad, así que cada millón de inferencias cuesta $9.48 en lugar de $0.0514: unas 184 veces más. El costo lo domina la capacidad ociosa, no el cómputo por petición; por eso las optimizaciones con más impacto reducen horas encendidas y tamaño de instancia, no la latencia.

Lote horario en Fargate efímero: 35 s por corrida (se cobran 60 s), 608 corridas al mes: $0.5256/mes, $2.33 por millón.

## Optimizaciones

| Optimización | Ahorro |
|---|---|
| Separar MLflow de la API (t3.medium -> t3.small) | 50.1 % |
| Graviton (t3.small -> t4g.small) | 19.3 % |
| Apagar en horas sin servicio | 16.7 % |
| Las tres combinadas frente al Compose actual | 66.4 % |
| Lote horario efímero frente a Fargate siempre encendido | 98.6 % |

Diferencia de precio de México (Central) frente a US East (N. Virginia): t3.small +4.8 %, t3.medium +5.0 %, t4g.small +4.8 %, t4g.medium +5.1 %, c7g.medium +5.0 %.
