############################################################
## run_did_multiplegt_dyn.R
##
## - builds monthly panel from incidents_long_fractional.csv
## - recomputes time-varying treatment AFTER fractional splitting
##   based on (cvi_area, month) and program history
## - runs DIDmultiplegtDYN::did_multiplegt_dyn
## - writes panel + summary
##
## Usage in VS Terminal:
## & "C:\Program Files\R\R-4.5.2\bin\Rscript.exe" run_did_multiplegt_dyn.R
############################################################

library(DIDmultiplegtDYN)
library(dplyr)
library(readr)

incidents_file <- "incidents_long_fractional.csv"
panel_file     <- "did_panel_dyn.csv"
summary_file   <- "did_multiplegt_dyn_summary.txt"

# ----------------------------
# Program timeline constants
# ----------------------------
PROGRAM_START <- as.Date("2023-08-01")
WAYNE_END     <- as.Date("2025-01-31")
TEAM_START    <- as.Date("2025-07-01")
LIVE_START    <- as.Date("2025-07-01")

# ----------------------------
# Treatment history logic
# ----------------------------
# IMPORTANT: 'cvi_area' here is the stable geography unit.
# If you renamed Team Pursuit geography to "Wayne Metro/Team Pursuit", use that consistently.
is_treated <- function(area, date_month) {
  # area: character scalar
  # date_month: Date (month start)
  if (is.na(area) || area == "" || area == "Non-CVI") return(FALSE)

  # Nobody treated before program start
  if (date_month < PROGRAM_START) return(FALSE)

  # Live in Peace treated only from 2025-07-01
  if (area == "Live in Peace") return(date_month >= LIVE_START)

  # Wayne Metro/Team Pursuit geography: treated Aug 2023 - Jan 2025; gap Feb-Jun 2025; treated again from Jul 2025
  if (area %in% c("Wayne Metro", "Team Pursuit", "Wayne Metro/Team Pursuit")) {
    if (date_month >= PROGRAM_START && date_month <= as.Date("2025-01-01")) return(TRUE)
    if (date_month >= as.Date("2025-02-01") && date_month < TEAM_START) return(FALSE)
    if (date_month >= TEAM_START) return(TRUE)
    return(FALSE)
  }

  # All other CVI areas treated from program start
  return(TRUE)
}

# ----------------------------
# 1) Load data
# ----------------------------
inc <- read_csv(incidents_file, col_types = cols())

# Accept either cvi_area or cvi_zone in the long file
geo_col <- if ("cvi_area" %in% names(inc)) "cvi_area" else if ("cvi_zone" %in% names(inc)) "cvi_zone" else NA_character_
if (is.na(geo_col)) {
  stop("Missing required geography column: expected 'cvi_area' or 'cvi_zone' in incidents file.")
}

req <- c("event_time", "source", geo_col, "weight")
missing <- setdiff(req, names(inc))
if (length(missing) > 0) {
  stop(paste0(
    "Missing required columns in incidents file: ",
    paste(missing, collapse = ", "),
    "\nMake sure incidents_long_fractional.csv includes event_time, source, cvi_area/cvi_zone, weight."
  ))
}

inc <- inc %>%
  mutate(
    event_time = as.Date(event_time),
    month      = as.Date(cut(event_time, "month")),
    cvi_area   = .data[[geo_col]],
    cvi_area   = if_else(is.na(cvi_area) | cvi_area == "", "Non-CVI", cvi_area),
    is_crime   = source == "RMS",
    is_call    = source != "RMS"
  ) %>%
  filter(!is.na(event_time)) %>%
  filter(event_time >= as.Date("2021-01-01"))

# ----------------------------
# 2) Build monthly panel
# ----------------------------
# Aggregate weighted counts.
panel <- inc %>%
  group_by(cvi_area, month) %>%
  summarise(
    crime = sum(weight[is_crime], na.rm = TRUE),
    calls = sum(weight[is_call],  na.rm = TRUE),
    .groups = "drop"
  )

# Create time index
panel <- panel %>%
  arrange(month, cvi_area)

time_levels <- sort(unique(panel$month))
panel <- panel %>%
  mutate(
    group = cvi_area,
    time  = match(month, time_levels)
  )

# Recompute time-varying treatment at the group-month level
panel <- panel %>%
  mutate(
    treated_share = as.numeric(mapply(is_treated, cvi_area, month)),
    treatment     = as.integer(treated_share > 0.5)  # kept to match your prior script pattern
  )

# ----------------------------
# 3) Diagnostics
# ----------------------------
cat("\n=== Checking panel structure ===\n")
print(panel %>% count(cvi_area))

cat("\nTime range:\n")
print(range(panel$time))

cat("\nTreatment switching by cvi_area (based on R history logic):\n")
print(
  panel %>%
    group_by(cvi_area) %>%
    summarise(
      min_treatment = min(treatment, na.rm = TRUE),
      max_treatment = max(treatment, na.rm = TRUE),
      avg_treated_share = mean(treated_share, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    arrange(cvi_area)
)

write_csv(panel, panel_file)
cat("[run_did_multiplegt_dyn] Saved panel to", panel_file, "\n")

# ----------------------------
# 4) Run did_multiplegt_dyn
# ----------------------------
reg <- did_multiplegt_dyn(
  df        = panel,
  outcome   = "crime",
  group     = "group",
  time      = "time",
  treatment = "treatment",
  effects   = 12,
  placebo   = 6,
  controls  = "calls",
  cluster   = "group",
  graph_off = TRUE
)

cat("[run_did_multiplegt_dyn] Model estimated.\n")

capture.output(
  {
    cat("=== did_multiplegt_dyn summary ===\n\n")
    print(summary(reg))
  },
  file = summary_file
)

cat("[run_did_multiplegt_dyn] Wrote summary to", summary_file, "\n")
