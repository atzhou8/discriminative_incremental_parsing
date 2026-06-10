library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(brms)
library(stringr)
source('./util.R')


# Rscript fits_dir/ out_dir/out.csv
args <- commandArgs(trailingOnly = TRUE)
fits_dir <-  args[1] 
out_dir <- args[2]

# Collect results in an R data.frame (one row per model file)
results <- data.frame(
  model = character(),
  NPS = numeric(),
  NPZ = numeric(),
  MVRR = numeric(),
  NPS_CI_low = numeric(),
  NPS_CI_high = numeric(),
  NPZ_CI_low = numeric(),
  NPZ_CI_high = numeric(),
  MVRR_CI_low = numeric(),
  MVRR_CI_high = numeric(),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

# find all .rds fit files under the provided directory
fit_files <- list.files(fits_dir, pattern = "\\.rds$", full.names = TRUE, recursive = TRUE)


for (f in fit_files) {
  message("Processing: ", f)
  model_fit <- readRDS(f)

  model_name <- basename(f)

  # Extract fixed effects mean
  fe <- fixef(model_fit)[, "Estimate"]
  nps <-  fe["AMBUAMB"] + fe["AMBUAMB:SZM1"]
  npz <- fe["AMBUAMB"] + fe["AMBUAMB:SZM2"]
  mvrr <- fe["AMBUAMB"]

  # Extract variances Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y)
  vc <- vcov(model_fit)
  nps_var <- vc["AMBUAMB", "AMBUAMB"] + vc["AMBUAMB:SZM1", "AMBUAMB:SZM1"] + 2 * vc["AMBUAMB", "AMBUAMB:SZM1"]
  npz_var <- vc["AMBUAMB", "AMBUAMB"] + vc["AMBUAMB:SZM2", "AMBUAMB:SZM2"] + 2 * vc["AMBUAMB", "AMBUAMB:SZM2"]
  mvrr_var <- vc["AMBUAMB", "AMBUAMB"]


  # Extract 95% CI (2.5% in each tail)
  z <- qnorm(0.975)
  nps_ci_low <- nps - z * sqrt(nps_var)
  nps_ci_high <- nps + z * sqrt(nps_var)
  npz_ci_low <- npz - z * sqrt(npz_var)
  npz_ci_high <- npz + z * sqrt(npz_var)
  mvrr_ci_low <- mvrr - z * sqrt(mvrr_var)
  mvrr_ci_high <- mvrr + z * sqrt(mvrr_var)

  results <- rbind(results, data.frame(
    model = model_name,
    NPS = nps,
    NPS_CI_low = nps_ci_low,
    NPS_CI_high = nps_ci_high,
    NPZ = npz,
    NPZ_CI_low = npz_ci_low,
    NPZ_CI_high = npz_ci_high,
    MVRR = mvrr,
    MVRR_CI_low = mvrr_ci_low,
    MVRR_CI_high = mvrr_ci_high,
    stringsAsFactors = FALSE
  ))

  rm(model_fit)
  gc()
}

write.csv(results, file = out_dir, row.names = FALSE)
