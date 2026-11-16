# Insumo técnico: recomendación y entrenamiento continuo

## Decisión provisional

Mantener los dos champion v1 sin modificar alias. Para bicicletas, la v2 reentrenada en enero-octubre mejora AP de 0.262416 a 0.277382 y recall de 0.817452 a 0.835085, pero precision baja de 0.118485 a 0.117357 y los falsos positivos aumentan de 100,714 a 104,008. La [comparación](ab_testing_summary.md) es retrospectiva y no tiene intervalos de confianza ni costos operativos. Reentrenar un candidato y promoverlo son decisiones distintas: **no promover v2 automáticamente**.

En el siguiente ciclo, preparar un candidato con datos recientes y cobertura anual, manteniendo congelada la definición de etiquetas y evitando ventanas que crucen particiones. Seleccionar configuración y umbral en un periodo de validación anterior, y evaluar ambos modelos en un periodo posterior independiente. Antes de promover, acordar con operación la métrica principal, límites mínimos de recall y presupuesto de alertas; revisar calibración y costos. Conservar v1 y su artefacto para rollback y requerir aprobación humana.

Resolver la discrepancia de desempate en 0.5 entre evaluación y API descrita en la [comparación](ab_testing_summary.md) antes de usar los conteos como estimaciones exactas de alertas.

## Continuous Training propuesto, no implementado

- Monitorear mensualmente calidad, disponibilidad de categorías, distribuciones, prevalencia y métricas cuando las etiquetas estén disponibles. El código actual está fijado a ventanas de 2025 y debe parametrizarse y probarse antes de operar mensualmente.
- Revisar trimestralmente la conveniencia de entrenar un candidato, sujeto a datos recientes suficientes y cobertura estacional. No activar entrenamiento o promoción automáticos por calendario.
- Abrir una investigación si una variable crítica presenta categorías fuera de train o si la prevalencia cambia al menos 20 % relativo frente a una referencia estacional comparable durante dos cortes mensuales consecutivos. Es un umbral inicial propuesto, no validado.
- Investigar si AP o precision caen al menos 10 % relativo frente a una referencia comparable durante dos cortes, con tamaños y etiquetas suficientes; revisar también recall y carga de alertas. No atribuir automáticamente la caída al drift.
- Detener el pipeline candidato si fallan integridad, hashes, disponibilidad o compatibilidad de predictores/etiquetas. Drift o p-values por sí solos no deben disparar promoción.

## Riesgos y pasos previos

Las etiquetas son proxies basadas en un catálogo de 2026 aplicado retrospectivamente a 2025. No representan inventario observado. Hay estacionalidad, ventanas solapadas y dependencia por estación, además de baja precision. No hay evidencia de un CT operativo, métricas de negocio, costos de intervención ni validación prospectiva. Definir comparaciones estacionales y bootstrap por bloques, seguridad de datos, trazabilidad de artefactos y monitoreo de calidad antes de automatizar.
