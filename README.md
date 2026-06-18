# Génesis de los datos semisintéticos para la detección de fraude

Este repositorio contiene el pipeline en R para construir una base semisintética de seguros de automóvil a partir de datos reales, inyectar fraude de forma controlada y generar los archivos de salida que alimentan un proyecto en Python basado en una arquitectura tabular y topológica para la detección de fraude.

## Flujo del proyecto

1. Se parte de una base real de seguros.
2. Se limpian y transforman variables.
3. Se prepara la tipología de siniestros.
4. Se construye la base semisintética intermedia.
5. Se inyecta fraude sintético con reglas de negocio y estructura de anillos.
6. Se exportan los archivos finales para el proyecto en Python.

## Archivos principales

- [Generador_Ranger.R](Generador_Ranger.R): prepara la base, ajusta la tipología de siniestros e imputa clases mediante Random Forest y reglas de negocio.
- [InyeccionFraude.R](InyeccionFraude.R): toma la base semisintética y genera la inyección de fraude, así como las estructuras relacionales finales.

## Archivos de entrada

- [Motor vehicle insurance data.csv](Motor%20vehicle%20insurance%20data.csv): base original de pólizas.
- [sample type claim.csv](sample%20type%20claim.csv): tipologías de siniestros.
- [Motor vehicle insurance full.rds](Motor%20vehicle%20insurance%20full.rds): base intermedia semisintética utilizada por el script de inyección.

## Salidas generadas

El pipeline genera tres archivos para el modelo en Python:

- claims.csv
- providers.csv
- edges.csv

## Nota sobre la reproducibilidad del generador Ranger

El archivo [Generador_Ranger.R](Generador_Ranger.R) no puede reejecutarse de forma equivalente al original. Por un error de planeación no se conservó la semilla que produjo [Motor vehicle insurance full.rds](Motor%20vehicle%20insurance%20full.rds) y volver a ejecutar ese paso alteraría la lógica manual con la que se construyeron los anillos de colusión. Por ese motivo, este script debe entenderse como parte del proceso histórico de construcción de la base, no como un paso completamente reproducible.

## Objetivo

El objetivo final es disponer de una base semisintética coherente para entrenar y evaluar modelos de detección de fraude desde dos perspectivas:

- tabular, para variables agregadas y estructurales
- topológica, para relaciones entre siniestros y proveedores.

