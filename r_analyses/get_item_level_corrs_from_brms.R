library(lme4)
library(tidyr)
library(stringr)
library(posterior)
library(tidybayes)
library(tidyverse)
library(dplyr)
library(brms)

args <- commandArgs(trailingOnly = TRUE)
# Rscript results/brms_eoi_models/mergedRT/roi/ results/correlations/xxx_raw.csv results/correlations/xxx_corr.csv
fits_dir <-  args[1] 
items_out_dir <- args[2]
corr_out_dir <- args[3]
empirical_dir <- 'results/brms_eoi_models/mergedRT/roi/empirical.rds'

compute_itemwise_posterior <- function(posterior_samp) {
  # For each posterior sample compute the item-wise effect implied by that sample
  get_item_matrix <- function(cols) {
    as.matrix(posterior_samp[, cols, drop = FALSE])
  }
  
  cn <- colnames(posterior_samp)
  r_intercept_cols <- grep('r_item.*Intercept', cn, value = TRUE)
  r_amb_nps_cols <- grep('r_item.*AMBUAMB:SZM1', cn, value = TRUE)
  r_amb_npz_cols <- grep('r_item.*AMBUAMB:SZM2', cn, value = TRUE)
  r_amb_cols <- setdiff(grep('r_item.*AMBUAMB', cn, value = TRUE), c(r_amb_nps_cols, r_amb_npz_cols))
  r_nps_cols <- setdiff(grep('r_item.*SZM1', cn, value = TRUE), r_amb_nps_cols)
  r_npz_cols <- setdiff(grep('r_item.*SZM2', cn, value = TRUE), r_amb_npz_cols)
  
  r_intercept <- get_item_matrix(r_intercept_cols)
  r_amb <- get_item_matrix(r_amb_cols)
  r_nps <- get_item_matrix(r_nps_cols)
  r_npz <- get_item_matrix(r_npz_cols)
  r_amb_nps <- get_item_matrix(r_amb_nps_cols)
  r_amb_npz <- get_item_matrix(r_amb_npz_cols)
  
  ncols <- ncol(posterior_samp)
  b_intercept <- posterior_samp[['b_Intercept']]
  b_amb <- posterior_samp[['b_AMBUAMB']]
  b_nps <- posterior_samp[['b_SZM1']]
  b_npz <- posterior_samp[['b_SZM2']]
  b_amb_nps <- posterior_samp[['b_AMBUAMB:SZM1']]
  b_amb_npz <- posterior_samp[['b_AMBUAMB:SZM2']]
  
  unamb_intercept <- sweep(r_intercept, 1, b_intercept, '+')
  unamb_NPS <- sweep(r_intercept + r_nps, 1, b_intercept + b_nps, '+')
  unamb_NPZ <- sweep(r_intercept + r_npz, 1, b_intercept + b_npz, '+')
  
  amb_intercept <- sweep(r_intercept + r_amb, 1, b_intercept + b_amb, '+')
  amb_NPS <- sweep(r_intercept + r_nps + r_amb + r_amb_nps, 1, b_intercept + b_nps + b_amb + b_amb_nps, '+')
  amb_NPZ <- sweep(r_intercept + r_npz + r_amb + r_amb_npz, 1, b_intercept + b_npz + b_amb + b_amb_npz, '+')
  
  # MVRR - AMBUAMB effect
  posterior_samp[, (1 + ncols):(24 + ncols)] <- amb_intercept - unamb_intercept
  
  # NPS - full SZM1 condition
  posterior_samp[, (25 + ncols):(48 + ncols)] <- amb_NPS - unamb_NPS
  
  # NPS - full SZM2 condition
  posterior_samp[, (49 + ncols):(72 + ncols)] <- amb_NPZ - unamb_NPZ
  
  pred_colnames <- c(
    paste0('GPE_MVRR_item', 1:24),
    paste0('GPE_NPS_item', 1:24),
    paste0('GPE_NPZ_item', 1:24)
  )
  colnames(posterior_samp)[(ncols + 1):(ncols + 72)] <- pred_colnames
  
  return(posterior_samp)
}

sample_model_correlations <- function(sampled_correlations, emp_draws, pred_draws, model_label, n_samples = 10000) {
  for (i in 1:n_samples) {
    # draw a single posterior row index for empirical and predicted draws
    n_emp_rows <- nrow(emp_draws)
    n_pred_rows <- nrow(pred_draws)
    idx_emp <- sample.int(n_emp_rows, 1)
    idx_pred <- sample.int(n_pred_rows, 1)

    item_samples <- do.call(rbind, lapply(c('MVRR', 'NPS', 'NPZ'), function(gp_type) {
      cols <- paste0('GPE_', gp_type, '_item', 1:24)
      emp_vec <- as.numeric(emp_draws[idx_emp, cols])
      pred_vec <- as.numeric(pred_draws[idx_pred, cols])
      data.frame(
        item = 1:24,
        type = gp_type,
        emp = emp_vec,
        predicted = pred_vec,
        stringsAsFactors = FALSE
      )
    }))

    aggregate_corr <- suppressWarnings(cor(item_samples$emp, item_samples$predicted, use = 'complete.obs'))

    sampled_correlations <- rbind(
      sampled_correlations,
      do.call(rbind, lapply(c('NPS', 'NPZ', 'MVRR'), function(gp_type) {
        curr <- subset(item_samples, type == gp_type)
        data.frame(
          correlation = suppressWarnings(cor(curr$emp, curr$predicted, use = 'complete.obs')),
          model = model_label,
          type = gp_type,
          stringsAsFactors = FALSE
        )
      })),
      data.frame(
        correlation = aggregate_corr,
        model = model_label,
        type = 'aggregate',
        stringsAsFactors = FALSE
      )
    )
  }

  sampled_correlations
}

# Save both posteriors and sampled correlations
eois_posterior <- data.frame(
  model=character(),
  type=character(),
  item=integer(),
  mean=numeric(),
  SE=numeric(),
  upper=numeric(),
  lower=numeric(),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
sampled_correlations <- data.frame(
  correlation=numeric(),
  model=character(),
  type=character(),
  stringsAsFactors = FALSE
)

# Reference distribution
emp_fit <- readRDS(empirical_dir)
posterior <- posterior_samples(emp_fit)
emp_posterior <- compute_itemwise_posterior(posterior)

# Now get correlations per model
models <- list.files(fits_dir, pattern = '.*\\.rds$', full.names = TRUE)
for (model in models) {
  model_basename <- basename(model)
  model_name <- sub('\\.rds$', '', model_basename)
  eoi_model <- try(readRDS(model), silent = TRUE)

  posterior <- posterior_samples(eoi_model)
  model_posterior <- compute_itemwise_posterior(posterior)

  sampled_correlations <- sample_model_correlations(
    sampled_correlations = sampled_correlations,
    emp_draws = emp_posterior,
    pred_draws = model_posterior,
    model_label = model_name
  )
  
  for (gp_type in c('NPS', 'NPZ', 'MVRR')) {
    items <- 1:24
    cols <- paste0('GPE_', gp_type, '_item', items)
    values <- model_posterior[, cols, drop = FALSE]
    eois_posterior <- rbind(
      eois_posterior,
      data.frame(
        model = model_name,
        type = gp_type,
        item = items,
        mean = colMeans(values),
        SE = apply(values, 2, sd),
        upper = apply(values, 2, quantile, probs = 0.975),
        lower = apply(values, 2, quantile, probs = 0.025)
      )
    )
  }

}
write.csv(eois_posterior, file = items_out_dir, row.names = FALSE)
write.csv(sampled_correlations, file = corr_out_dir, row.names = FALSE)
