library(dplyr)
library(ranger)
library(lubridate)

# Cargar los datos de la cartera de pólizas y la tabla de tipología de siniestros
datos <- read.csv("Motor vehicle insurance data.csv", sep = ";", stringsAsFactors = FALSE)
tipos <- read.csv("sample type claim.csv", sep = ";", stringsAsFactors = FALSE)


# ==============================================================================
# 1. CONVERSIÓN DE FORMATOS DE FECHA Y VARIABLES DERIVADAS
# ==============================================================================

# Declarar las columnas temporales que deben estandarizarse a tipo Date
date_columns <- c("Date_start_contract", "Date_last_renewal", "Date_next_renewal", 
                  "Date_birth", "Date_driving_licence", "Date_lapse")

# Convertir las columnas de fecha desde texto al formato Día/Mes/Año
datos[date_columns] <- lapply(datos[date_columns], as.Date, format = "%d/%m/%Y")

datos <- datos %>%
  mutate(
    # Derivar el año de referencia a partir de la última renovación de la póliza
    Current_Year     = year(Date_last_renewal),
    # Calcular la edad del conductor en el año de referencia
    Driver_Age       = Current_Year - year(Date_birth),
    # Calcular los años de experiencia al volante desde la obtención del carnet
    Experience_Years = Current_Year - year(Date_driving_licence),
    # Calcular la antigüedad del vehículo, asignando 1 año a los matriculados en el año en curso
    Vehicle_Age      = if_else(Current_Year == Year_matriculation, 1, Current_Year - Year_matriculation),
    
    # Recodificar el tipo de vehículo a factor con etiquetas legibles
    Type_risk = factor(Type_risk, 
                       levels = c(1, 2, 3, 4), 
                       labels = c("Motorbike", "Van", "Passenger car", "Agricultural")),
    
    # Recodificar la zona geográfica a factor (Urbana / Rural)
    Area = factor(Area, 
                  levels = c(0, 1), 
                  labels = c("Urban", "Rural")),
    
    # Recodificar la modalidad de pago a factor (Anual / Semestral)
    Payment = factor(Payment,
                     levels = c(0, 1),
                     labels = c("Annual", "Semiannual")),
    
    # Recodificar la presencia de segundo conductor a factor
    Second_driver = factor(Second_driver,
                           levels = c(0, 1),
                           labels = c("Unique", "Multiple")),
    
    # Imputar gasolina como combustible por defecto en motos sin dato (regla de negocio)
    Type_fuel = ifelse(Type_risk == "Motorbike" & is.na(Type_fuel), "P", Type_fuel),
    
    # Recodificar el tipo de combustible a factor tras la imputación
    Type_fuel = factor(Type_fuel, 
                       levels = c("P", "D"), 
                       labels = c("Petrol", "Diesel")),
    
    # Imputar una longitud estándar de 2 metros en motos sin dato registrado
    Length = ifelse(Type_risk == "Motorbike" & is.na(Length), 2.0, Length)
  )

# Imputar los valores nulos restantes de Length con la media por tipo de vehículo
datos <- datos %>%
  group_by(Type_risk) %>%
  mutate(
    Length = ifelse(is.na(Length), 
                    round(mean(Length, na.rm = TRUE), 2), 
                    Length)
) %>%
  ungroup()   # Desagrupar para evitar que el group_by afecte a operaciones posteriores


# ==============================================================================
# 2. PREPARACIÓN DE LA TABLA DE TIPOLOGÍA DE SINIESTROS
# ==============================================================================

# Consolidar un único registro por ID seleccionando la tipología del tramo más costoso
tipos_limpio <- tipos %>% 
  group_by(ID) %>% 
  summarise(
    # Conservar el coste total anual del siniestro (constante dentro del ID) para el cruce posterior
    Cost_claims_year = unique(Cost_claims_year),
    # Seleccionar la etiqueta asociada al subconcepto de mayor coste reclamado
    Claims_type = Claims_type[which.max(Cost_claims_by_type)],
    .groups = 'drop'
  ) %>%
  # Redondear el coste a dos decimales para garantizar coincidencia exacta en el join
  mutate(Cost_claims_year = round(Cost_claims_year, 2))


# ==============================================================================
# 3. CRUCE DE DATOS Y FILTRADO DE LA POBLACIÓN DE SINIESTROS
# ==============================================================================

datos_join <- datos %>%
  # Retener únicamente las pólizas que han registrado al menos un siniestro
  filter(Cost_claims_year > 0) %>%
  # Incorporar la tipología consolidada mediante la clave compuesta ID + coste anual
  left_join(tipos_limpio, by = c("ID", "Cost_claims_year"))


# ==============================================================================
# 4. IMPUTACIÓN MULTIVARIANTE MEDIANTE RANDOM FOREST
# ==============================================================================

# 4.1. Segmentar el conjunto en población etiquetada y población a imputar

# Aislar los siniestros con tipología conocida para entrenar el clasificador
train_imputation <- datos_join %>% filter(!is.na(Claims_type))

# Aislar los siniestros sin tipología que serán objeto de imputación
to_predict <- datos_join %>% filter(is.na(Claims_type))


# 4.2. Entrenar un Random Forest probabilístico para predecir la tipología
# Usar probability = TRUE para obtener distribuciones completas y preservar la varianza
modelo_imputacion <- ranger(
  as.factor(Claims_type) ~ Cost_claims_year + 
    Driver_Age + Experience_Years + N_doors +
    Vehicle_Age + Value_vehicle + Power + 
    Cylinder_capacity + Weight + Length + 
    Seniority + Premium + N_claims_history,
  data = train_imputation,
  num.trees = 500,              # Fijar 500 árboles para estabilizar las predicciones
  probability = TRUE,
  importance = "impurity"       # Registrar la importancia de variables para diagnóstico
)

# Inspeccionar la importancia relativa de las variables explicativas
modelo_imputacion$variable.importance

# Obtener la matriz de probabilidades predichas (filas = siniestros, columnas = categorías)
predicciones_prob <- predict(modelo_imputacion, data = to_predict)$predictions


# ==============================================================================
# 5. PRE-CÁLCULO DE UMBRALES ECONÓMICOS POR CATEGORÍA DE BAJO COSTE
# ==============================================================================
# Objetivo: impedir que el sorteo asigne categorías baratas (other, theft,
# travel assistance, broken windows) a siniestros con coste atípicamente alto.
# Se emplea el criterio robusto mediana + 3·IQR por tratarse de distribuciones
# sesgadas a la derecha, y se acota además por el máximo histórico observado
# para no superar nunca lo visto empíricamente en la cartera.

categorias_bajo_coste <- c("other", "theft", "travel assistance", "broken windows")

umbrales <- train_imputation %>%
  filter(Claims_type %in% categorias_bajo_coste) %>%
  group_by(Claims_type) %>%
  summarise(
    mediana    = median(Cost_claims_year),
    iqr        = IQR(Cost_claims_year),
    # Calcular el umbral robusto frente a outliers
    limite_iqr = mediana + 3 * iqr,
    # Registrar el máximo observado como tope duro de seguridad
    limite_max = max(Cost_claims_year),
    .groups    = "drop"
  ) %>%
  # Adoptar como umbral final el más restrictivo de los dos criterios
  mutate(limite = pmin(limite_iqr, limite_max))

# Convertir los umbrales a vector nombrado para acceso rápido dentro del bucle
tope <- setNames(umbrales$limite, umbrales$Claims_type)


# ==============================================================================
# 6. SORTEO DE LA TIPOLOGÍA IMPUTADA CON REGLAS DE NEGOCIO
# ==============================================================================
# Recorrer cada siniestro a imputar y muestrear una categoría respetando:
#   (a) el filtro de coherencia económica sobre categorías de bajo coste
#   (b) el filtro estructural que prohíbe "broken windows" en motocicletas

to_predict$Claims_type <- sapply(1:nrow(to_predict), function(i) {
  
  # Extraer la distribución de probabilidad y el catálogo de categorías para el siniestro i
  probs <- predicciones_prob[i, ]
  opciones <- colnames(predicciones_prob)
  
  # ----------------------------------------------------------------------------
  # Filtro (a): coherencia económica
  # ----------------------------------------------------------------------------
  # Anular la probabilidad de cualquier categoría cuyo umbral sea superado
  # por el coste anual del siniestro, evitando imputaciones inverosímiles.
  coste_i <- to_predict$Cost_claims_year[i]
  
  for (cat in names(tope)) {
    if (coste_i > tope[[cat]]) {
      idx <- which(opciones == cat)
      if (length(idx) > 0) probs[idx] <- 0
    }
  }
  
  # Red de seguridad: si el filtro económico ha eliminado todas las opciones,
  # restablecer una distribución uniforme para permitir el muestreo.
  if (sum(probs) == 0) {
    probs <- rep(1, length(probs))
  }
  
  # ----------------------------------------------------------------------------
  # Filtro (b): exclusión de "broken windows" en motocicletas
  # ----------------------------------------------------------------------------
  # Las motocicletas carecen de lunas asegurables, por lo que esta categoría
  # debe quedar descartada sea cual sea la predicción del modelo.
  if (to_predict$Type_risk[i] == "Motorbike") {
    
    # Localizar la columna "broken windows" (búsqueda insensible a mayúsculas)
    idx_lunas <- grep("broken windows", opciones, ignore.case = TRUE)
    
    if (length(idx_lunas) > 0) {
      # Anular definitivamente la probabilidad de lunas rotas
      probs[idx_lunas] <- 0 
    }
    
    # Red de seguridad: si lunas era la única opción viable, repartir
    # equiprobabilidad entre el resto manteniendo lunas en cero.
    if (sum(probs) == 0) {
      probs <- rep(1, length(probs))
      probs[idx_lunas] <- 0 
    }
  }
  
  # Muestrear la categoría final de acuerdo con las probabilidades corregidas
  sample(opciones, size = 1, prob = probs)
})


# ==============================================================================
# 7. CONSOLIDACIÓN Y EXPORTACIÓN DE LA BASE DE DATOS DEFINITIVA
# ==============================================================================

# Unir la población etiquetada con la población recién imputada
datos_final <- bind_rows(train_imputation, to_predict)

# Ordenar los registros por cliente y cronológicamente por fechas de contrato
datos_final <- datos_final %>%
  arrange(ID, Date_start_contract, Date_last_renewal)

# Verificar que no persisten valores nulos en la variable objetivo tras la imputación
cat("Valores nulos en Claims_type tras la imputación:", sum(is.na(datos_final$Claims_type)), "\n")

# Excluir vehículos agrícolas del conjunto final y descartar variables no utilizadas
datos_final <- datos_final %>% 
  filter(Type_risk != "Agricultural") %>% 
  select(-N_claims_year, -R_Claims_history)

# Exportar el conjunto consolidado en formato binario (rds) y texto plano (csv)
# saveRDS(datos_final, "Motor vehicle insurance full.rds")
# write.csv(datos_final, "Motor vehicle insurance full.csv", row.names = FALSE)