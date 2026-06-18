# ============================================================
# INYECCIÓN DE FRAUDE EN DATASET DE SEGUROS DE AUTO
# Pipeline completo: 9 pasos, desde dataset base hasta 3 CSVs
# (claims.csv, providers.csv, edges.csv) listos para GNN en Python.
# ============================================================


# ============================================================
# LIBRERÍAS Y CONFIGURACIÓN GLOBAL
# ============================================================

suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
  library(readr)
  library(tidyr)
  library(purrr)
})

set.seed(2025)


# ============================================================
# PASO 0 — PREPARACIÓN DEL DATASET BASE
# ============================================================
# Entrada : RDS semisintético de seguros de automóvil.
# Salida  : df limpio con 8 columnas de fraude inicializadas
#           a sus valores por defecto (sin fraude asignado).

df <- readRDS("Motor vehicle insurance full.rds")
df$Experience_Years <- pmax(df$Experience_Years, 0)
cat("Dataset original:", nrow(df), "filas\n")

# Columnas eliminadas por diseño previo:
#   N_claims_year    → ambigua; no indica reclamaciones del mismo tipo
#   Lapse/Date_lapse → ruido sin fechas de siniestro
#   R_Claims_history → derivada de N_claims_history, redundante
df <- df %>% select(-any_of(c("Lapse", "Date_lapse",
                              "N_claims_year", "R_Claims_history")))

# Inicializar columnas de fraude con valores por defecto
df <- df %>% mutate(
  Event_ID              = NA_character_,  # Identificador de siniestro (único o compartido en multichoques)
  is_fraud              = FALSE,          # Etiqueta binaria objetivo del modelo
  Fraud_type            = "legitimate",   # legitimate / organized / opportunistic
  Ring_ID               = "none",         # Pertenencia a anillo: A, B, C, lobo_1/2/3 o none
  Provider_workshop_ID  = NA_character_,  # Taller asignado al siniestro
  Provider_clinic_ID    = NA_character_,  # Clínica asignada (injuries y algunos negligence/other)
  Provider_lawyer_ID    = NA_character_,  # Abogado asignado (complaint, injuries, negligence, etc.)
  Linked_to_fraud_event = FALSE           # TRUE solo para víctimas legítimas del T3A
)

cat("Tras limpieza:", nrow(df), "filas,", ncol(df), "columnas\n\n")


# ============================================================
# PASO 1 — CATÁLOGO DE PROVEEDORES
# ============================================================
# Genera los 3 tipos de proveedores (talleres, clínicas, abogados)
# con IDs legibles, marca de fraude, anillo de pertenencia y
# Fraud_ratio individual sorteado en [0.35, 0.65].
# Los primeros n_fraud IDs de cada tipo son los fraudulentos.

# Genera un tibble de proveedores con su configuración completa.
make_providers <- function(prefix, n, n_fraud, ring_assign) {
  ids         <- sprintf("%s_%03d", prefix, 1:n)
  is_fraud    <- c(rep(TRUE, n_fraud), rep(FALSE, n - n_fraud))
  ring        <- c(ring_assign, rep("none", n - n_fraud))
  fraud_ratio <- c(runif(n_fraud, 0.35, 0.65), rep(NA_real_, n - n_fraud))
  
  tibble(
    Provider_ID   = ids,
    Provider_type = prefix,
    Is_fraudulent = is_fraud,
    Ring_ID       = ring,
    Fraud_ratio   = fraud_ratio
  )
}

# Talleres: 11 fraudulentos — 3 en A, 3 en B, 2 en C, 1 por cada lobo solitario
workshop_rings <- c(rep("A", 3), rep("B", 3), rep("C", 2),
                    "lobo_1", "lobo_2", "lobo_3")
workshops <- make_providers("WSH", 300, 11, workshop_rings)

# Clínicas: 3 fraudulentas — 2 en A (especialista en lesiones), 1 en B
clinic_rings <- c(rep("A", 2), "B")
clinics <- make_providers("CLN", 30, 3, clinic_rings)

# Abogados: 6 fraudulentos — 3 en A, 2 en B, 1 en C
lawyer_rings <- c(rep("A", 3), rep("B", 2), "C")
lawyers <- make_providers("LAW", 80, 6, lawyer_rings)

# Catálogo unificado de los 3 tipos
providers <- bind_rows(workshops, clinics, lawyers)

cat("Proveedores creados:\n")
cat("  Talleres:", nrow(workshops), "(", sum(workshops$Is_fraudulent), "fraude)\n")
cat("  Clínicas:", nrow(clinics),   "(", sum(clinics$Is_fraudulent),   "fraude)\n")
cat("  Abogados:", nrow(lawyers),   "(", sum(lawyers$Is_fraudulent),   "fraude)\n\n")

# Extrae los IDs de proveedores de un tipo y anillo concretos.
get_ring_providers <- function(prov_type, ring) {
  providers %>%
    filter(Provider_type == prov_type, Ring_ID == ring) %>%
    pull(Provider_ID)
}

# Pools de proveedores fraudulentos por anillo y tipo
WSH_A    <- get_ring_providers("WSH", "A")
WSH_B    <- get_ring_providers("WSH", "B")
WSH_C    <- get_ring_providers("WSH", "C")
CLN_A    <- get_ring_providers("CLN", "A")
CLN_B    <- get_ring_providers("CLN", "B")
LAW_A    <- get_ring_providers("LAW", "A")
LAW_B    <- get_ring_providers("LAW", "B")
LAW_C    <- get_ring_providers("LAW", "C")

WSH_lobo <- providers %>%
  filter(Ring_ID %in% c("lobo_1", "lobo_2", "lobo_3")) %>%
  pull(Provider_ID)

# Pools de proveedores totalmente legítimos
WSH_legit <- providers %>% filter(Provider_type == "WSH", !Is_fraudulent) %>% pull(Provider_ID)
CLN_legit <- providers %>% filter(Provider_type == "CLN", !Is_fraudulent) %>% pull(Provider_ID)
LAW_legit <- providers %>% filter(Provider_type == "LAW", !Is_fraudulent) %>% pull(Provider_ID)

# Matriz de probabilidades de asignación de proveedores por tipo de siniestro.
# Es la única fuente de verdad del script: toda asignación (legítima u organizada)
# debe respetar estas probabilidades.
reglas <- data.frame(
  Claims_type = c("injuries", "complaint", "negligence", "broken windows",
                  "all risks", "fire", "theft", "other", "travel assistance"),
  p_wsh = c(0.40, 0.70, 0.80, 1.00, 1.00, 1.00, 0.80, 0.60, 0.15),
  p_cln = c(1.00, 0.10, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
  p_law = c(0.80, 1.00, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
  stringsAsFactors = FALSE
)
rownames(reglas) <- reglas$Claims_type

cat("Reglas de negocio cargadas:\n")
print(reglas)
cat("\n")


# ============================================================
# PASO 2 — INYECCIÓN DE LOS 32 MULTICHOQUES COORDINADOS
# ============================================================
# Asigna un Event_ID compartido a dos clientes (A y B) que participan
# en el mismo siniestro coordinado. Según el tipo de multichoque,
# uno o ambos pueden ser fraudulentos; en T3A, el cliente B es víctima.
#
# Entrada : df con columnas de fraude inicializadas.
# Salida  : df con Event_ID, is_fraud, Ring_ID y proveedores
#           asignados en los 32 pares de multichoques.

# Inyecta un par de filas como participantes de un mismo siniestro.
# fraud_a / fraud_b controlan si cada cliente es fraudulento.
# linked_b = TRUE marca al cliente B como víctima legítima (solo T3A).
assign_event <- function(df, event_id, id_a, year_a, id_b, year_b,
                         ring, wsh_a, cln_a, law_a, wsh_b, cln_b, law_b,
                         fraud_a = TRUE, fraud_b = TRUE, linked_b = FALSE) {
  
  idx_a <- which(df$ID == id_a & df$Current_Year == year_a)
  idx_b <- which(df$ID == id_b & df$Current_Year == year_b)
  stopifnot(length(idx_a) == 1, length(idx_b) == 1)
  
  df$Event_ID[c(idx_a, idx_b)] <- event_id
  
  # Cliente A
  if (fraud_a) {
    df$is_fraud[idx_a]   <- TRUE
    df$Fraud_type[idx_a] <- "organized"
    df$Ring_ID[idx_a]    <- ring
  }
  df$Provider_workshop_ID[idx_a] <- wsh_a
  if (!is.na(cln_a)) df$Provider_clinic_ID[idx_a] <- cln_a
  if (!is.na(law_a)) df$Provider_lawyer_ID[idx_a] <- law_a
  
  # Cliente B (puede ser víctima legítima si linked_b = TRUE)
  if (fraud_b) {
    df$is_fraud[idx_b]   <- TRUE
    df$Fraud_type[idx_b] <- "organized"
    df$Ring_ID[idx_b]    <- ring
  }
  if (linked_b) df$Linked_to_fraud_event[idx_b] <- TRUE
  df$Provider_workshop_ID[idx_b] <- wsh_b
  if (!is.na(cln_b)) df$Provider_clinic_ID[idx_b] <- cln_b
  if (!is.na(law_b)) df$Provider_lawyer_ID[idx_b] <- law_b
  
  df
}

# Lista estática de los 32 multichoques.
# Formato por elemento: list(tipo, año, id_cliente_A, id_cliente_B)
# IDs seleccionados manualmente respetando Area, año, Claims_type y Type_risk.
multichoques <- list(
  # T1 — "Choque simulado para reparar daños viejos"
  # Anillo C · broken windows + all risks · ambos turismos · mismo taller · sin clínica ni abogado
  list("T1", 2015, 33440,  1964), list("T1", 2016,  1540, 13072),
  list("T1", 2016, 37725, 52131), list("T1", 2016, 30588, 30627),
  list("T1", 2016, 24752, 49754), list("T1", 2017, 39804,  5677),
  list("T1", 2017, 21648, 34856), list("T1", 2017,    83, 41990),
  list("T1", 2018, 21909, 37706), list("T1", 2018, 53359,  1245),
  
  # T2 — "Alcance trasero con latigazo simulado"
  # Anillo A · A = injuries (taller + clínica + abogado) · B = all risks (solo taller)
  list("T2", 2016, 29202, 47010), list("T2", 2016, 42107, 38432),
  list("T2", 2016, 21736, 42136), list("T2", 2017, 30437,  4727),
  list("T2", 2017, 24408,  3039), list("T2", 2017, 20525,  4144),
  list("T2", 2018, 34058, 49113), list("T2", 2018, 38840,  1834),
  
  # T3A — "Moto fantasma" · motorista fraudulento (Anillo A) + turismo víctima legítima
  list("T3A", 2017,  1432,  7122), list("T3A", 2017,  8280, 20536),
  list("T3A", 2017, 33759, 19178), list("T3A", 2018, 28234, 43508),
  list("T3A", 2018, 50655,  5273),
  
  # T3B — "Turismo vs furgoneta, fraude patrimonial"
  # Anillo B · ambos solo taller · reparación inflada
  list("T3B", 2016, 15222, 23694), list("T3B", 2016, 30507, 34522),
  list("T3B", 2016, 34399, 48499), list("T3B", 2016, 51593, 35078),
  list("T3B", 2016, 24401, 50207), list("T3B", 2016, 38427, 33711),
  list("T3B", 2017, 51770, 50758), list("T3B", 2017, 52205, 33870),
  list("T3B", 2017, 51721, 47459)
)

# Sortear proveedores garantizando reparto mínimo:
# cada proveedor del anillo recibe al menos 1 evento antes de completar aleatoriamente.

# T1: 10 eventos repartidos entre los 2 talleres del Anillo C (5 + 5)
t1_wsh <- sample(c(rep(WSH_C[1], 5), rep(WSH_C[2], 5)))

# T2: 8 eventos con taller + clínica + abogado del Anillo A
t2_wsh <- sample(c(WSH_A, sample(WSH_A, 5, replace = TRUE)))
t2_cln <- sample(c(CLN_A, sample(CLN_A, 6, replace = TRUE)))
t2_law <- sample(c(LAW_A, sample(LAW_A, 5, replace = TRUE)))

# T3A: 5 eventos del Anillo A (motorista) + 5 talleres legítimos (víctima)
t3a_wsh       <- sample(c(WSH_A, sample(WSH_A, 2, replace = TRUE)))
t3a_cln       <- sample(c(CLN_A, sample(CLN_A, 3, replace = TRUE)))
t3a_law       <- sample(c(LAW_A, sample(LAW_A, 2, replace = TRUE)))
t3a_legit_wsh <- sample(WSH_legit, 5)

# T3B: 9 eventos entre los 3 talleres del Anillo B
t3b_wsh <- sample(c(WSH_B, sample(WSH_B, 7, replace = TRUE)))

# Contadores de índice por tipo y contador global de eventos
i_t1 <- 0; i_t2 <- 0; i_t3a <- 0; i_t3b <- 0; event_counter <- 0

for (m in multichoques) {
  event_counter <- event_counter + 1
  event_id <- sprintf("siniestro_%05d", event_counter)
  tp <- m[[1]]; yr <- m[[2]]; id_a <- m[[3]]; id_b <- m[[4]]
  
  if (tp == "T1") {
    i_t1 <- i_t1 + 1
    df <- assign_event(df, event_id, id_a, yr, id_b, yr,
                       ring  = "C",
                       wsh_a = t1_wsh[i_t1], cln_a = NA, law_a = NA,
                       wsh_b = t1_wsh[i_t1], cln_b = NA, law_b = NA)
    
  } else if (tp == "T2") {
    i_t2 <- i_t2 + 1
    df <- assign_event(df, event_id, id_a, yr, id_b, yr,
                       ring  = "A",
                       wsh_a = t2_wsh[i_t2], cln_a = t2_cln[i_t2], law_a = t2_law[i_t2],
                       wsh_b = t2_wsh[i_t2], cln_b = NA,            law_b = NA)
    
  } else if (tp == "T3A") {
    i_t3a <- i_t3a + 1
    # Cliente B es víctima legítima: taller legítimo propio, sin etiqueta de fraude
    df <- assign_event(df, event_id, id_a, yr, id_b, yr,
                       ring  = "A",
                       wsh_a = t3a_wsh[i_t3a],       cln_a = t3a_cln[i_t3a], law_a = t3a_law[i_t3a],
                       wsh_b = t3a_legit_wsh[i_t3a],  cln_b = NA,             law_b = NA,
                       fraud_a = TRUE, fraud_b = FALSE, linked_b = TRUE)
    
  } else if (tp == "T3B") {
    i_t3b <- i_t3b + 1
    df <- assign_event(df, event_id, id_a, yr, id_b, yr,
                       ring  = "B",
                       wsh_a = t3b_wsh[i_t3b], cln_a = NA, law_a = NA,
                       wsh_b = t3b_wsh[i_t3b], cln_b = NA, law_b = NA)
  }
}

cat("Eventos multichoque inyectados:", event_counter, "\n")
cat("Filas marcadas como fraude:",     sum(df$is_fraud),              "\n")
cat("Filas Linked_to_fraud_event:",    sum(df$Linked_to_fraud_event), "\n\n")


# ============================================================
# PASO 3 — FRAUDE ORGANIZADO SOLITARIO (~215 FILAS)
# ============================================================
# Selecciona candidatos mediante una puntuación de riesgo (behav_score)
# y los etiqueta como fraude organizado asignándoles proveedores del anillo.
#
# Entrada : df con multichoques inyectados.
# Salida  : df con ~215 filas adicionales de fraude organizado.

# behav_score (0–3): acumula 1 punto por cada señal de riesgo presente
df <- df %>% mutate(
  cost_rank = ave(Cost_claims_year, Claims_type,
                  FUN = function(x) rank(x) / length(x)),
  behav_score =
    as.numeric(Seniority <= 3 & Premium >= quantile(Premium, 0.6, na.rm = TRUE)) +
    as.numeric(N_claims_history >= quantile(N_claims_history, 0.75, na.rm = TRUE)) +
    as.numeric(cost_rank >= 0.80)
)

# Devuelve índices de filas elegibles para inyección de fraude.
# Si el filtro estricto (behav_score >= min_score) no da suficientes candidatos,
# relaja el score manteniendo Claims_type y Area.
# Deduplica por ID: un cliente solo puede aportar una fila al muestreo.
pick_candidates <- function(df, n, claims_types, areas = NULL, min_score = 1) {
  mask <- is.na(df$Event_ID) & !df$is_fraud & !df$Linked_to_fraud_event &
    df$Claims_type %in% claims_types & df$behav_score >= min_score
  if (!is.null(areas)) mask <- mask & df$Area %in% areas
  idx <- which(mask)
  
  if (length(idx) < n) {
    warning(sprintf("pick_candidates: relajando filtro behav_score para %s",
                    paste(claims_types, collapse = ",")))
    mask2 <- is.na(df$Event_ID) & !df$is_fraud & !df$Linked_to_fraud_event &
      df$Claims_type %in% claims_types
    if (!is.null(areas)) mask2 <- mask2 & df$Area %in% areas
    idx <- which(mask2)
  }
  
  idx <- idx[!duplicated(df$ID[idx])]
  sample(idx, size = min(n, length(idx)))
}

# Etiqueta las filas como fraude organizado y asigna proveedores del anillo.
# El taller siempre al 100% (mecanismo central de la trama y señal topológica clave).
# Clínica y abogado se asignan según las probabilidades de la matriz `reglas`.
# Si el anillo no tiene clínica/abogado propios, se usa el pool legítimo como fallback.
assign_solo <- function(df, idx, ring, wsh_pool, cln_pool = NULL, law_pool = NULL) {
  n <- length(idx)
  df$is_fraud[idx]   <- TRUE
  df$Fraud_type[idx] <- "organized"
  df$Ring_ID[idx]    <- ring
  df$Event_ID[idx]   <- sprintf("siniestro_solo_%s_%05d", ring, seq_len(n))
  
  for (i in seq_along(idx)) {
    fila <- idx[i]
    tipo <- df$Claims_type[fila]
    r    <- reglas[tipo, ]
    
    df$Provider_workshop_ID[fila] <- sample(wsh_pool, 1)
    
    if (r$p_cln > 0 && runif(1) <= r$p_cln) {
      pool_cln <- if (!is.null(cln_pool)) cln_pool else CLN_legit
      df$Provider_clinic_ID[fila] <- sample(pool_cln, 1)
    }
    
    if (r$p_law > 0 && runif(1) <= r$p_law) {
      pool_law <- if (!is.null(law_pool)) law_pool else LAW_legit
      df$Provider_lawyer_ID[fila] <- sample(pool_law, 1)
    }
  }
  
  df
}

# Anillo A "Lesiones": 65 filas en injuries/negligence urbanas
idx_A <- pick_candidates(df, 65, c("injuries", "negligence"), areas = "Urban", min_score = 1)
df    <- assign_solo(df, idx_A, "A", WSH_A, CLN_A, LAW_A)

# Anillo B "Colisiones": 55 filas en negligence/all risks/complaint urbanas
idx_B <- pick_candidates(df, 55, c("negligence", "all risks", "complaint"), areas = "Urban", min_score = 1)
df    <- assign_solo(df, idx_B, "B", WSH_B, CLN_B, LAW_B)

# Anillo C "Carrocería": 45 filas mixtas
# Sin clínica propia; el abogado se asignará legítimo si aplica por la lógica de assign_solo
idx_C <- pick_candidates(df, 45, c("all risks", "broken windows", "theft", "complaint"), min_score = 1)
df    <- assign_solo(df, idx_C, "C", WSH_C, NULL, LAW_C)

# Lobos solitarios: 17 filas cada uno, solo taller del anillo correspondiente
for (k in 1:3) {
  idx_l <- pick_candidates(df, 17, c("all risks", "negligence", "broken windows"), min_score = 1)
  df    <- assign_solo(df, idx_l, paste0("lobo_", k), WSH_lobo[k])
}

cat("Paso 3 completado. Fraude organizado solitario:", sum(df$Fraud_type == "organized") - 59, "filas\n")
cat("Total fraude:", sum(df$is_fraud), "\n\n")


# ============================================================
# PASO 4 — FRAUDE OPORTUNISTA (~413 FILAS)
# ============================================================
# Fraude individual sin coordinación de anillo. Se distinguen dos subtipos:
#   (fp) con taller fraudulento cómplice
#   (lp) solo el asegurado actúa fraudulentamente; taller legítimo o ausente

# Pool de talleres fraudulentos de todos los anillos
WSH_fraud_all <- providers %>%
  filter(Provider_type == "WSH", Is_fraudulent) %>%
  pull(Provider_ID)

# Asigna taller, clínica y abogado a un conjunto de filas
# según las probabilidades de la matriz `reglas`.
assign_providers_by_rules <- function(df, idx, wsh_pool, cln_pool, law_pool) {
  for (i in seq_along(idx)) {
    fila <- idx[i]
    tipo <- df$Claims_type[fila]
    r    <- reglas[tipo, ]
    
    if (r$p_wsh > 0 && runif(1) <= r$p_wsh)
      df$Provider_workshop_ID[fila] <- sample(wsh_pool, 1)
    if (r$p_cln > 0 && runif(1) <= r$p_cln)
      df$Provider_clinic_ID[fila]   <- sample(cln_pool, 1)
    if (r$p_law > 0 && runif(1) <= r$p_law)
      df$Provider_lawyer_ID[fila]   <- sample(law_pool, 1)
  }
  df
}

# --- Paso 4a: 6 casos injuries con proveedores legítimos ---
# injuries: WSH = 0.40, CLN = 1.00, LAW = 0.80
idx <- pick_candidates(df, 6, c("injuries"), min_score = 0)
df$is_fraud[idx]   <- TRUE
df$Fraud_type[idx] <- "opportunistic"
df$Event_ID[idx]   <- sprintf("siniestro_opp_inj_%05d", seq_along(idx))
for (i in seq_along(idx)) {
  fila <- idx[i]
  if (runif(1) <= 0.40) df$Provider_workshop_ID[fila] <- sample(WSH_legit, 1)
  df$Provider_clinic_ID[fila] <- sample(CLN_legit, 1)
  if (runif(1) <= 0.80) df$Provider_lawyer_ID[fila]   <- sample(LAW_legit, 1)
}

# Tipos de siniestro elegibles para el fraude oportunista general
opp_types <- c("all risks", "broken windows", "negligence",
               "complaint", "theft", "other", "fire")

# --- Paso 4a (cont.): 200 oportunistas con taller fraudulento cómplice ---
# Taller fraudulento al 100% (mecanismo de colaboración);
# clínica y abogado según las reglas del tipo de siniestro.
idx <- pick_candidates(df, 200, opp_types, min_score = 1)
df$is_fraud[idx]             <- TRUE
df$Fraud_type[idx]           <- "opportunistic"
df$Event_ID[idx]             <- sprintf("siniestro_opp_fp_%05d", seq_along(idx))
df$Provider_workshop_ID[idx] <- sample(WSH_fraud_all, length(idx), replace = TRUE)
for (i in seq_along(idx)) {
  fila <- idx[i]
  tipo <- df$Claims_type[fila]
  r    <- reglas[tipo, ]
  if (r$p_cln > 0 && runif(1) <= r$p_cln)
    df$Provider_clinic_ID[fila] <- sample(CLN_legit, 1)
  if (r$p_law > 0 && runif(1) <= r$p_law)
    df$Provider_lawyer_ID[fila] <- sample(LAW_legit, 1)
}

cat("Total fraude:", sum(df$is_fraud),
    "| Ratio:", round(mean(df$is_fraud) * 100, 2), "%\n")

# --- Paso 4a (cont.): 207 oportunistas con proveedor 100% legítimo ---
# Señal de fraude solo en perfil del cliente y coste; el taller no es cómplice.
# El prefijo opp_lp es necesario para el filtro del Paso 4c.

set.seed(2025)
n_attractors <- 25  # 25 talleres concentran los 207 opp_lp
WSH_attractors <- sample(WSH_legit, n_attractors)

idx <- pick_candidates(df, 207, opp_types, min_score = 1)
df$is_fraud[idx]             <- TRUE
df$Fraud_type[idx]           <- "opportunistic"
df$Event_ID[idx]             <- sprintf("siniestro_opp_lp_%05d", seq_along(idx))
df$Provider_workshop_ID[idx] <- sample(WSH_attractors, length(idx), replace = TRUE) #df$Provider_workshop_ID[idx] <- sample(WSH_legit, length(idx), replace = TRUE)

# --- Paso 4b: fraude oportunista en travel assistance (25 claims) ---
# Elimina la señal artificial de "travel assistance = legítimo garantizado".
# 13 con taller fraudulento cómplice (grúas/remolques inflados)
idx <- pick_candidates(df, 13, c("travel assistance"), min_score = 0)
df$is_fraud[idx]             <- TRUE
df$Fraud_type[idx]           <- "opportunistic"
df$Event_ID[idx]             <- sprintf("siniestro_opp_ta_fp_%05d", seq_along(idx))
df$Provider_workshop_ID[idx] <- sample(WSH_fraud_all, length(idx), replace = TRUE)

# 12 con taller legítimo (asegurado que simula avería; taller no cómplice)
idx <- pick_candidates(df, 12, c("travel assistance"), min_score = 0)
df$is_fraud[idx]             <- TRUE
df$Fraud_type[idx]           <- "opportunistic"
df$Event_ID[idx]             <- sprintf("siniestro_opp_ta_lp_%05d", seq_along(idx))
df$Provider_workshop_ID[idx] <- sample(WSH_legit, length(idx), replace = TRUE)

cat("Fraude travel assistance añadido: 25 claims\n")
cat("Total fraude tras Paso 4b:", sum(df$is_fraud),
    "| Ratio:", round(mean(df$is_fraud) * 100, 2), "%\n\n")

# --- Paso 4c: eliminar taller en ~35 fraudes oportunistas (lp) ---
# Solo en tipos donde un claim sin taller tiene sentido de negocio:
#   complaint  → reclamación legal pura, sin reparación física
#   negligence → daños menores, solo peritaje y reclamación
#   injuries   → lesión sin daño al vehículo (atropello, latigazo)
# NUNCA para broken windows, all risks, fire, theft ni travel assistance.
set.seed(2025)  # Re-seed local para reproducibilidad de este sub-paso

tipos_sin_taller <- c("complaint", "negligence", "injuries")

idx_opp_lp_elegible <- which(
  df$is_fraud &
    df$Fraud_type == "opportunistic" &
    grepl("opp_lp", df$Event_ID) &
    !is.na(df$Provider_workshop_ID) &
    df$Claims_type %in% tipos_sin_taller
)

n_sin_taller   <- min(35, length(idx_opp_lp_elegible))
idx_sin_taller <- sample(idx_opp_lp_elegible, n_sin_taller)
df$Provider_workshop_ID[idx_sin_taller] <- NA

# Garantizar que cada claim sin taller conserva al menos clínica o abogado
for (fila in idx_sin_taller) {
  tipo <- df$Claims_type[fila]
  r    <- reglas[tipo, ]
  if (r$p_cln > 0 && is.na(df$Provider_clinic_ID[fila]))
    df$Provider_clinic_ID[fila] <- sample(CLN_legit, 1)
  if (r$p_law > 0 && is.na(df$Provider_lawyer_ID[fila]))
    df$Provider_lawyer_ID[fila] <- sample(LAW_legit, 1)
}

cat("Fraudes sin taller:", n_sin_taller, "\n")
cat("  Tipos afectados:\n")
print(table(df$Claims_type[idx_sin_taller]))
cat("  Total fraudes con taller:", sum(df$is_fraud & !is.na(df$Provider_workshop_ID)),
    "de", sum(df$is_fraud), "\n")
cat("  % fraudes con taller:", round(
  sum(df$is_fraud & !is.na(df$Provider_workshop_ID)) / sum(df$is_fraud) * 100, 1), "%\n\n")

# --- Paso 4d: eliminar taller en ~20% del fraude de theft ---
# Robo total = sin reparación; el claim queda como nodo aislado en el grafo.
# Es realista: robo total denunciado a policía no pasa por taller ni abogado.
set.seed(2025)  # Seed específica para este sub-paso

idx_theft_fraud <- which(
  df$is_fraud &
    df$Claims_type == "theft" &
    !is.na(df$Provider_workshop_ID)
)

n_quitar <- round(0.20 * length(idx_theft_fraud))
if (n_quitar > 0) {
  idx_quitar <- sample(idx_theft_fraud, n_quitar)
  df$Provider_workshop_ID[idx_quitar] <- NA
  cat("Theft fraudulento sin taller (robo total):", n_quitar,
      "de", length(idx_theft_fraud), "\n\n")
} else {
  cat("Theft fraudulento sin taller: 0 (insuficientes candidatos)\n\n")
}


# ============================================================
# PASO 5 — ASIGNAR PROVEEDORES A SINIESTROS LEGÍTIMOS
# ============================================================
# Asigna proveedores legítimos a los claims que aún no tienen proveedor,
# usando reglas_legit (versión adaptada de `reglas` para controlar
# la esparsidad del grafo sin inundar de aristas legítimas).
#
# Entrada : df con todo el fraude inyectado.
# Salida  : df con Provider_*_ID rellenados en claims legítimos.

# Asigna proveedores al subconjunto de filas que cumplan `mask`,
# muestreando de forma independiente con las probabilidades dadas.
assign_legit <- function(df, mask, p_wsh, p_cln, p_law) {
  n <- sum(mask)
  if (p_wsh > 0) {
    sel <- sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(p_wsh, 1 - p_wsh))
    df$Provider_workshop_ID[which(mask)[sel]] <- sample(WSH_legit, sum(sel), replace = TRUE)
  }
  if (p_cln > 0) {
    sel <- sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(p_cln, 1 - p_cln))
    df$Provider_clinic_ID[which(mask)[sel]]   <- sample(CLN_legit, sum(sel), replace = TRUE)
  }
  if (p_law > 0) {
    sel <- sample(c(TRUE, FALSE), n, replace = TRUE, prob = c(p_law, 1 - p_law))
    df$Provider_lawyer_ID[which(mask)[sel]]   <- sample(LAW_legit, sum(sel), replace = TRUE)
  }
  df
}

# Probabilidades para legítimos: equilibran coherencia de negocio y esparsidad del grafo
reglas_legit <- data.frame(
  Claims_type = c("injuries", "complaint", "negligence", "broken windows",
                  "all risks", "fire", "theft", "other", "travel assistance"),
  p_wsh = c(1.00, 0.30, 0.95, 1.00, 1.00, 1.00, 0.60, 0.60, 0.15),
  p_cln = c(1.00, 0.00, 0.10, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
  p_law = c(0.60, 1.00, 0.60, 0.00, 0.10, 0.10, 0.20, 0.00, 0.00),
  stringsAsFactors = FALSE
)
rownames(reglas_legit) <- reglas_legit$Claims_type

# Aplicar a claims legítimos sin proveedor asignado aún
legit_base <- !df$is_fraud & !df$Linked_to_fraud_event & is.na(df$Provider_workshop_ID)
for (tipo in reglas_legit$Claims_type) {
  r    <- reglas_legit[tipo, ]
  mask <- legit_base & df$Claims_type == tipo
  df   <- assign_legit(df, mask, r$p_wsh, r$p_cln, r$p_law)
}

# --- Paso 5b: multichoques legítimos (35 pares) ---
# Añade pares legítimos con Event_ID compartido para diluir la señal en
# aristas claim_claim_event del ~92% al ~44% de fraude. Los claims
# conservan los proveedores legítimos ya asignados; is_fraud permanece FALSE.
n_pares_legit <- 35

cand_mc <- which(!df$is_fraud & !df$Linked_to_fraud_event &
                   is.na(df$Event_ID) &
                   !is.na(df$Provider_workshop_ID))

set.seed(2025)  # Sub-seed reproducible para el emparejamiento
pares <- data.frame(a = integer(0), b = integer(0))

for (yr in sort(unique(df$Current_Year[cand_mc]))) {
  yr_idx   <- cand_mc[df$Current_Year[cand_mc] == yr]
  if (length(yr_idx) < 2) next
  shuffled <- sample(yr_idx)
  n_take   <- floor(length(shuffled) / 2)
  for (i in seq_len(n_take)) {
    pares <- rbind(pares, data.frame(a = shuffled[2 * i - 1],
                                     b = shuffled[2 * i]))
  }
}

pares <- pares[seq_len(min(n_pares_legit, nrow(pares))), ]

for (i in seq_len(nrow(pares))) {
  eid <- sprintf("siniestro_legit_mc_%05d", i)
  df$Event_ID[pares$a[i]] <- eid
  df$Event_ID[pares$b[i]] <- eid
}

cat("Multichoques legítimos añadidos:", nrow(pares), "pares\n")
cat("  Claims legítimos en co_event:", 2 * nrow(pares), "\n")

# Verificación de la distribución de fraude en aristas co_event
temp_mc <- df[!is.na(df$Event_ID), ] %>%
  group_by(Event_ID) %>%
  filter(n() > 1) %>%
  ungroup()
cat("  Total claims en multichoques:", nrow(temp_mc), "\n")
cat("  De los cuales fraude:",          sum(temp_mc$is_fraud), "\n")
cat("  Tasa fraude co_event:",          round(mean(temp_mc$is_fraud) * 100, 1), "%\n\n")

# Asignar Event_ID individual correlativo a los siniestros sin evento aún
idx_noev <- which(is.na(df$Event_ID))
df$Event_ID[idx_noev] <- sprintf("siniestro_ind_%06d", seq_along(idx_noev))


# ============================================================
# PASO 5.5 — REDISTRIBUCIÓN PARA RESPETAR FRAUD_RATIO
# ============================================================
# Sin este paso, los proveedores fraudulentos recibirían solo claims
# fraudulentos (tasa 100%), lo cual es irreal. Un proveedor cómplice
# mantiene una cartera mixta como tapadera.
#
# Reasigna claims legítimos (ya asignados a proveedores legítimos)
# a proveedores fraudulentos hasta que cada uno alcance su Fraud_ratio
# objetivo (sorteado en [0.35, 0.65] en el Paso 1).
# Solo cambia el proveedor asignado; is_fraud permanece FALSE.
#
# Se aplica a los 3 tipos de proveedor: workshop, clinic, lawyer.

redistribute_to_fraud_providers <- function(df, providers, provider_type, id_column) {
  fraud_provs <- providers %>%
    filter(Provider_type == provider_type, Is_fraudulent) %>%
    select(Provider_ID, Fraud_ratio)
  
  for (k in seq_len(nrow(fraud_provs))) {
    pid          <- fraud_provs$Provider_ID[k]
    target_ratio <- fraud_provs$Fraud_ratio[k]
    
    n_fraud_here <- sum(df[[id_column]] == pid & df$is_fraud, na.rm = TRUE)
    if (n_fraud_here == 0) next
    
    n_total_target <- round(n_fraud_here / target_ratio)
    n_legit_needed <- n_total_target - n_fraud_here
    if (n_legit_needed <= 0) next
    
    candidates <- which(
      !df$is_fraud &
        !df$Linked_to_fraud_event &
        !is.na(df[[id_column]]) &
        df[[id_column]] != pid
    )
    
    if (length(candidates) < n_legit_needed) {
      warning(sprintf("%s %s: solo hay %d candidatos para %d necesarios",
                      provider_type, pid, length(candidates), n_legit_needed))
      n_legit_needed <- length(candidates)
    }
    
    idx_reassign <- sample(candidates, n_legit_needed)
    df[[id_column]][idx_reassign] <- pid
    
    cat(sprintf("  %s: %d fraudes + %d legitimos (objetivo %.2f, real %.2f)\n",
                pid, n_fraud_here, n_legit_needed, target_ratio,
                n_fraud_here / (n_fraud_here + n_legit_needed)))
  }
  
  df
}

cat("\nRedistribuyendo talleres fraudulentos (Fraud_ratio):\n")
df <- redistribute_to_fraud_providers(df, providers, "WSH", "Provider_workshop_ID")
cat("\nRedistribuyendo clinicas fraudulentas:\n")
df <- redistribute_to_fraud_providers(df, providers, "CLN", "Provider_clinic_ID")
cat("\nRedistribuyendo abogados fraudulentos:\n")
df <- redistribute_to_fraud_providers(df, providers, "LAW", "Provider_lawyer_ID")

# Crear claim_id aquí para que esté disponible en el Safety Net (Paso 6)
df <- df %>%
  mutate(claim_id = sprintf("CLM_%06d", row_number())) %>%
  relocate(claim_id)


# ============================================================
# PASO 6 (SAFETY NET) — VALIDACIÓN DE COHERENCIA
# ============================================================
# Detecta y repara inconsistencias antes de la exportación:
#   Regla 1: tipos obligatorios deben tener al menos un proveedor asignado
#   Regla 2: ningún claim debe tener un proveedor donde la matriz marca p = 0
#   Regla 3: resumen de nodos aislados y verificación de tipos permitidos
#
# Nota: el "Paso 6" de análisis descriptivo fue omitido del script de producción.

cat("========== SAFETY NET: Validación de coherencia ==========\n")

# Regla 1: asignar proveedor obligatorio a claims huérfanos
obligatorios <- list(
  "injuries"       = "Provider_clinic_ID",
  "broken windows" = "Provider_workshop_ID",
  "all risks"      = "Provider_workshop_ID",
  "fire"           = "Provider_workshop_ID",
  "complaint"      = "Provider_lawyer_ID",
  "negligence"     = "Provider_workshop_ID"
)

n_reparados <- 0
for (tipo in names(obligatorios)) {
  col  <- obligatorios[[tipo]]
  pool <- switch(col,
                 "Provider_workshop_ID" = WSH_legit,
                 "Provider_clinic_ID"   = CLN_legit,
                 "Provider_lawyer_ID"   = LAW_legit)
  
  huerfanos <- which(
    df$Claims_type == tipo &
      is.na(df$Provider_workshop_ID) &
      is.na(df$Provider_clinic_ID) &
      is.na(df$Provider_lawyer_ID)
  )
  
  if (length(huerfanos) > 0) {
    df[[col]][huerfanos] <- sample(pool, length(huerfanos), replace = TRUE)
    cat(sprintf("  ⚠ %s: %d huérfanos reparados (asignado %s)\n",
                tipo, length(huerfanos), col))
    n_reparados <- n_reparados + length(huerfanos)
  }
}

if (n_reparados == 0) cat("  ✓ Ningún huérfano encontrado en tipos obligatorios\n")

# Regla 2: violaciones de la matriz (proveedor asignado donde p = 0)
n_violaciones <- 0
for (i in seq_len(nrow(df))) {
  tipo <- df$Claims_type[i]
  if (!(tipo %in% reglas$Claims_type)) next
  r <- reglas[tipo, ]
  
  if (r$p_cln == 0 && !is.na(df$Provider_clinic_ID[i])) {
    n_violaciones <- n_violaciones + 1
    if (n_violaciones <= 5)
      cat(sprintf("  ✗ %s (%s): tiene clínica pero p_cln=0\n", df$claim_id[i], tipo))
  }
  
  if (r$p_law == 0 && !is.na(df$Provider_lawyer_ID[i])) {
    n_violaciones <- n_violaciones + 1
    if (n_violaciones <= 5)
      cat(sprintf("  ✗ %s (%s): tiene abogado pero p_law=0\n", df$claim_id[i], tipo))
  }
}

if (n_violaciones == 0) {
  cat("  ✓ Ninguna violación de la matriz de reglas\n")
} else {
  cat(sprintf("  ✗ TOTAL VIOLACIONES: %d (mostrando primeras 5)\n", n_violaciones))
  cat("    Pueden originarse en el Paso 2 (multichoques hardcodeados) o en el Paso 5.5.\n")
}

# Regla 3: resumen de nodos aislados (sin ningún proveedor asignado)
aislados <- which(
  is.na(df$Provider_workshop_ID) &
    is.na(df$Provider_clinic_ID) &
    is.na(df$Provider_lawyer_ID)
)
cat(sprintf("\n  Nodos aislados totales: %d (%.1f%%)\n",
            length(aislados), length(aislados) / nrow(df) * 100))
cat("  Distribución por tipo:\n")
print(table(df$Claims_type[aislados]))

tipos_aislados_ok <- c("travel assistance", "theft", "other")
aislados_mal <- aislados[!(df$Claims_type[aislados] %in% tipos_aislados_ok)]
if (length(aislados_mal) > 0) {
  cat(sprintf("  ✗ %d nodos aislados de tipo NO permitido:\n", length(aislados_mal)))
  print(table(df$Claims_type[aislados_mal]))
} else {
  cat("  ✓ Todos los nodos aislados son de tipos permitidos\n")
}

cat("\n========== FIN SAFETY NET ==========\n\n")


# ============================================================
# PASO 7 — INFLADO DE COSTES EN FRAUDE
# ============================================================
# Multiplica Cost_claims_year por un factor aleatorio en un rango
# específico según Claims_type.
#
# Exclusiones (sin inflado):
#   - Multichoques Tipo 1 del Anillo C: fraude cualitativo, no económico
#   - Víctimas legítimas del T3A (Linked_to_fraud_event = TRUE)
#
# Se guarda Cost_claims_year_original para auditoría y comparación.

df$Cost_claims_year_original <- df$Cost_claims_year

# Infla las filas indicadas multiplicando por un factor uniform(factor_min, factor_max).
inflate_rows <- function(df, idx, factor_min, factor_max) {
  if (length(idx) == 0) return(df)
  factors <- runif(length(idx), factor_min, factor_max)
  df$Cost_claims_year[idx] <- df$Cost_claims_year[idx] * factors
  df
}

# Identificar exclusiones
is_coord    <- !is.na(df$Event_ID) &
  grepl("^siniestro_0+[0-9]+$", df$Event_ID) &
  !grepl("^siniestro_ind|^siniestro_solo|^siniestro_opp", df$Event_ID)
is_T1_fraud <- is_coord & df$Ring_ID == "C"
exclude     <- is_T1_fraud | df$Linked_to_fraud_event

# Inflado diferenciado por tipo de siniestro
idx <- which(df$is_fraud & df$Claims_type == "injuries" & !exclude)
df  <- inflate_rows(df, idx, 1.8, 2.2)

elig <- which(df$is_fraud & df$Claims_type == "all risks" & !exclude)
idx  <- sample(elig, size = round(0.85 * length(elig)))
df   <- inflate_rows(df, idx, 1.4, 2.0)

idx <- which(df$is_fraud & df$Claims_type == "theft" & !exclude)
df  <- inflate_rows(df, idx, 1.5, 2.0)

elig <- which(df$is_fraud & df$Claims_type == "negligence" & !exclude)
idx  <- sample(elig, size = round(0.90 * length(elig)))
df   <- inflate_rows(df, idx, 1.4, 1.8)

elig <- which(df$is_fraud & df$Claims_type == "complaint" & !exclude)
idx  <- sample(elig, size = round(0.85 * length(elig)))
df   <- inflate_rows(df, idx, 1.5, 2.0)

elig <- which(df$is_fraud & df$Claims_type == "broken windows" & !exclude)
idx  <- sample(elig, size = round(0.85 * length(elig)))
df   <- inflate_rows(df, idx, 1.3, 1.8)

elig <- which(df$is_fraud & df$Claims_type == "other" & !exclude)
idx  <- sample(elig, size = round(0.75 * length(elig)))
df   <- inflate_rows(df, idx, 1.3, 1.8)

# travel assistance: grúas/asistencias con facturación inflada (factor moderado)
elig <- which(df$is_fraud & df$Claims_type == "travel assistance" & !exclude)
idx  <- sample(elig, size = round(0.10 * length(elig)))
df   <- inflate_rows(df, idx, 1.3, 1.8)

total_inflated <- sum(df$Cost_claims_year != df$Cost_claims_year_original)
cat("Total filas infladas:", total_inflated,
    "(", round(100 * total_inflated / sum(df$is_fraud), 1), "% del fraude)\n")


# ============================================================
# PASO 8 — FEATURES AGREGADAS PARA EL GRAFO
# ============================================================
# Calcula estadísticos a nivel de proveedor y evento para enriquecer
# las features tabulares del modelo (XGBoost) sin necesitar el grafo.
# También añade LossRatio, Cost_per_value y z_score_type por Claims_type.

# Volumen y coste medio por taller
wsh_stats <- df %>%
  filter(!is.na(Provider_workshop_ID)) %>%
  group_by(Provider_workshop_ID) %>%
  summarise(Workshop_total_claims = n(),
            Workshop_avg_cost     = mean(Cost_claims_year, na.rm = TRUE),
            .groups = "drop")

# Volumen y coste medio por clínica
cln_stats <- df %>%
  filter(!is.na(Provider_clinic_ID)) %>%
  group_by(Provider_clinic_ID) %>%
  summarise(Clinic_total_claims = n(),
            Clinic_avg_cost     = mean(Cost_claims_year, na.rm = TRUE),
            .groups = "drop")

# Volumen y coste medio por abogado
law_stats <- df %>%
  filter(!is.na(Provider_lawyer_ID)) %>%
  group_by(Provider_lawyer_ID) %>%
  summarise(Lawyer_total_claims = n(),
            Lawyer_avg_cost     = mean(Cost_claims_year, na.rm = TRUE),
            .groups = "drop")

# Número de participantes por evento (1 = individual, 2 = multichoque)
event_stats <- df %>%
  group_by(Event_ID) %>%
  summarise(Event_participants_count = n(), .groups = "drop")

# Unir todas las features agregadas al dataset principal
df <- df %>%
  left_join(wsh_stats,   by = "Provider_workshop_ID") %>%
  left_join(cln_stats,   by = "Provider_clinic_ID")   %>%
  left_join(law_stats,   by = "Provider_lawyer_ID")   %>%
  left_join(event_stats, by = "Event_ID")

# Ratios de coste y z-score relativo al tipo de siniestro legítimo
df <- df %>% mutate(
  LossRatio      = Cost_claims_year / Premium,
  Cost_per_value = Cost_claims_year / Value_vehicle
)

# z-score calculado únicamente sobre la distribución de claims legítimos
stats_legit <- df %>%
  filter(!is_fraud, !Linked_to_fraud_event) %>%
  group_by(Claims_type) %>%
  summarise(
    mu_legit = mean(Cost_claims_year, na.rm = TRUE),
    sd_legit = sd(Cost_claims_year,   na.rm = TRUE),
    .groups  = "drop"
  )

df <- df %>%
  left_join(stats_legit, by = "Claims_type") %>%
  mutate(z_score_type = (Cost_claims_year - mu_legit) / pmax(sd_legit, 1)) %>%
  select(-mu_legit, -sd_legit)


# ============================================================
# PASO 9 — EXPORTACIÓN DE LOS 3 CSVs Y RESUMEN FINAL
# ============================================================
# Genera los 3 archivos de entrada para el pipeline de modelado en Python:
#   claims.csv    → nodos-siniestro con todas las features
#   providers.csv → nodos-proveedor con metadatos
#   edges.csv     → aristas del grafo (4 tipos)

write_csv(df, "claims.csv")
write_csv(providers, "providers.csv")

# Arista tipo 1: siniestro → taller
edges_wsh <- df %>%
  filter(!is.na(Provider_workshop_ID)) %>%
  transmute(source = claim_id, target = Provider_workshop_ID, edge_type = "claim_workshop")

# Arista tipo 2: siniestro → clínica
edges_cln <- df %>%
  filter(!is.na(Provider_clinic_ID)) %>%
  transmute(source = claim_id, target = Provider_clinic_ID, edge_type = "claim_clinic")

# Arista tipo 3: siniestro → abogado
edges_law <- df %>%
  filter(!is.na(Provider_lawyer_ID)) %>%
  transmute(source = claim_id, target = Provider_lawyer_ID, edge_type = "claim_lawyer")

# Arista tipo 4: siniestros que comparten Event_ID (solo multichoques).
# Self-join + filtro x < y para evitar duplicados y auto-bucles.
edges_event <- df %>%
  filter(Event_participants_count > 1) %>%
  select(claim_id, Event_ID) %>%
  inner_join(., ., by = "Event_ID", relationship = "many-to-many") %>%
  filter(claim_id.x < claim_id.y) %>%
  transmute(source = claim_id.x, target = claim_id.y, edge_type = "claim_claim_event")

edges <- bind_rows(edges_wsh, edges_cln, edges_law, edges_event)
write_csv(edges, "edges.csv")

cat("\n======= RESUMEN FINAL =======\n")
cat("claims.csv:", nrow(df), "filas,", ncol(df), "columnas\n")
cat("  Fraude total:", sum(df$is_fraud),
    "(", round(100 * mean(df$is_fraud), 2), "%)\n")
cat("  Organized:",     sum(df$Fraud_type == "organized"),     "\n")
cat("  Opportunistic:", sum(df$Fraud_type == "opportunistic"), "\n")
cat("  Legitimate:",    sum(df$Fraud_type == "legitimate"),    "\n")
cat("  Linked_to_fraud_event:", sum(df$Linked_to_fraud_event), "\n")
cat("providers.csv:", nrow(providers), "proveedores (",
    sum(providers$Is_fraudulent), "fraudulentos)\n")
cat("edges.csv:", nrow(edges), "aristas (",
    nrow(edges_wsh), "workshop,",
    nrow(edges_cln), "clinic,",
    nrow(edges_law), "lawyer,",
    nrow(edges_event), "event)\n")
cat("\nPipeline completado.\n")
