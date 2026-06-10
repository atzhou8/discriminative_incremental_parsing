library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(stringr)
library(lme4)
source('./util.R')

gp.spr <- load_data("ClassicGP")
print("Loaded ClassicGP data.")
gp.spr$SZM1 <- ifelse(gp.spr$CONSTRUCTION=="NPS",1,0)
gp.spr$SZM2 <- ifelse(gp.spr$CONSTRUCTION=="NPZ",1,0)
roi_metrics = read.csv(file.path('./results/rt_models/mergedRT', 'roi', 'metrics.csv'))
print(paste('Loaded ROI metrics from', file.path('./results/rt_models/mergedRT', 'roi', 'metrics.csv')))

roi_df <- bind_metrics(gp.spr, roi_metrics)
dir.create('results/EOIs', recursive=TRUE, showWarnings=FALSE)
dir.create('results/eoi_models/merged', recursive=TRUE, showWarnings=FALSE)

rds_files <- list.files(
    file.path('./results/rt_models/mergedRT', 'roi/eachword', ''),
  pattern = '\\.[Rr][Dd][Ss]$',
  full.names = TRUE,
  recursive = TRUE
)
rds_files <- sort(rds_files)
output_path <- 'results/EOIs/lm/roi_EOIs.csv'
# Collect results in an R data.frame (one row per model x ROI)
results <- data.frame(
  model = character(),
  ROI = integer(),
  NPS = numeric(),
  NPZ = numeric(),
  MVRR = numeric(),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

is_nonconverged <- function(fit) {
  if (inherits(fit, "try-error") || is.null(fit)) {
    return(TRUE)
  }
  FALSE
}
lmer_ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 200000))

# Get empirical
print("Fitting empirical model.")
model_fit <- try(
  lmer(RT_merged ~ AMBUAMB*(SZM1+SZM2) +
           (1 + AMBUAMB * (SZM1 + SZM2) || item) +
           (1 + AMBUAMB * (SZM1 + SZM2) || participant),
       data = subset(gp.spr, ROI == 0 & !is.na(RT_merged)),
       control = lmer_ctrl),
  silent = TRUE
)
saveRDS(model_fit, file = file.path('results/lm_eoi_models/mergedRT/roi/empirical.rds'))
message('Saved RDS for empirical')

if (is_nonconverged(model_fit)) {
  nps <- NaN
  npz <- NaN
  mvrr <- NaN
} else {
  nps <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM1"]
  npz <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM2"]
  mvrr <- fixef(model_fit)["AMBUAMB"]
}
results <- rbind(results, data.frame(
    model = 'empirical',
    ROI = 0,
    NPS = nps,
    NPZ = npz,
    MVRR = mvrr,
    stringsAsFactors = FALSE
  )
)
write.csv(results, file = output_path, row.names = FALSE)
print(paste("Wrote", nrow(results), "rows to", output_path))

for (rds_file in rds_files) {
  model_name <- str_remove(basename(rds_file), '\\.[Rr][Dd][Ss]$')
  message('Processing ', model_name)

  predicted <- Predicting_RT_with_spillover(roi_df, rds_file)
  # Fit the EOI model to the RT model's predictions, not the raw RTs.
  model_fit <- try(
    lmer(predicted ~ AMBUAMB*(SZM1+SZM2) +
           (1 + AMBUAMB * (SZM1 + SZM2) || item) +
           (1 + AMBUAMB * (SZM1 + SZM2) || participant),
         data = subset(predicted, ROI == 0 & !is.na(predicted)),
         control = lmer_ctrl),
    silent = TRUE
  )

  if (is_nonconverged(model_fit)) {
    nps <- NaN; npz <- NaN; mvrr <- NaN
  } else {
    nps <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM1"]
    npz <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM2"]
    mvrr <- fixef(model_fit)["AMBUAMB"]
    saveRDS(model_fit, file = file.path('results/lm_eoi_models/mergedRT/roi', paste0(model_name, '.rds')))
    message('Saved RDS for ', model_name)
  }

  results <- rbind(results, data.frame(
    model = model_name,
    ROI = 0,
    NPS = nps,
    NPZ = npz,
    MVRR = mvrr,
    stringsAsFactors = FALSE
  ))
  # Persist this model's metrics to CSV immediately (append)
  this_row <- data.frame(
    model = model_name,
    ROI = 0,
    NPS = nps,
    NPZ = npz,
    MVRR = mvrr,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  if (!file.exists(output_path)) {
    tryCatch({
      write.table(this_row, file = output_path, sep = ",", row.names = FALSE, col.names = TRUE)
      message('Created CSV and appended ', model_name)
    }, error = function(e) message('Failed to write CSV header for ', model_name, ': ', e$message))
  } else {
    tryCatch({
      write.table(this_row, file = output_path, sep = ",", row.names = FALSE, col.names = FALSE, append = TRUE)
      message('Appended ', model_name, ' to CSV')
    }, error = function(e) message('Failed to append ', model_name, ' to CSV: ', e$message))
  }

}
