library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(brms)
library(stringr)
source('./util.R')


ncores <- 4
num_threads <- 14
niter <- 12000
nwarmup <- 6000
seed <- 117
output_dir <- './results/brms_eoi_models/mergedRT'

print(paste('Cores:', ncores, 'Iter:', niter, 
            'Warmup:', nwarmup, 'Seed:', seed, 'Threading:', num_threads)
)

# Still bind metrics as a shortcut to get only garden paths
gp.spr <- load_data('ClassicGP')
gp.spr$SZM1 <- ifelse(gp.spr$CONSTRUCTION == 'NPS', 1, 0)
gp.spr$SZM2 <- ifelse(gp.spr$CONSTRUCTION == 'NPZ', 1, 0)
print('Loaded ClassicGP data.')

roi_metrics <- read.csv('./results/rt_models/mergedRT/roi/metrics.csv')
print('Loaded ROI metrics.')
roi_df <- bind_metrics(gp.spr, roi_metrics)

print('Fitting empirical brm model.')
roi <- 0
empirical_data <- subset(roi_df, ROI == roi & !is.na(RT_merged))
if (nrow(empirical_data) == 0) {
  stop('No usable rows for empirical model.')
}
prior <- c(
  prior('normal(900,2000)', class = 'Intercept'),
  prior('normal(0,450)', class = 'b'),
  prior('normal(0,300)', class = 'sd'),
  prior('normal(0,600)', class = 'sigma')
)
model_fit <- brm(
  RT_merged ~ AMBUAMB * (SZM1 + SZM2) +
    (1 + AMBUAMB * (SZM1 + SZM2) || item) +
    (1 | participant),
  data = empirical_data,
  iter = niter,
  warmup = nwarmup,
  cores = ncores,
  chains = ncores,
  threads = threading(num_threads),
  backend = 'cmdstanr',
  prior=prior,
  silent = FALSE
)

out_path <- paste0(output_dir, '/empirical.rds')
saveRDS(model_fit, file = out_path)
print(paste('Saved', out_path))

rm(model_fit, empirical_data)
gc()

print('Finished empirical fit.')
