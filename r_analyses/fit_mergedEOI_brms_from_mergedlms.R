library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(brms)
library(stringr)
library(cmdstanr)
source('./util.R')

# Rscript fit_mergedEOI_brms_from_mergedlms.R [model_idx] [rds_input_dir] [output_dir]
args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 3) {
  stop('Usage: Rscript fit_mergedEOI_brms_from_mergedlms.R <model_idx> <rds_input_dir> <output_dir>')
}

model_idx <- suppressWarnings(as.integer(args[1]))
if (is.na(model_idx)) {
  stop(paste0('model_idx must be an integer; got ', args[1]))
}

# Required CLI args (no defaults)
rds_input_dir <- args[2]
output_dir <- args[3]

ncores <- 4
num_threads <- 14
niter <- 16000
nwarmup <- 8000
maxtree <- 12
delta <- 0.80
seed <- 117

print(paste('Cores:', ncores, 'Iter:', niter, 
            'Warmup:', nwarmup, 'Seed:', seed, 'Threading', num_threads,
            'treedepth:', maxtree, 'adaptdelta', delta)
)

# Collating normalized data from training
gp.spr <- load_data('ClassicGP')
print('Loaded ClassicGP data.')

gp.spr$SZM1 <- ifelse(gp.spr$CONSTRUCTION == 'NPS', 1, 0)
gp.spr$SZM2 <- ifelse(gp.spr$CONSTRUCTION == 'NPZ', 1, 0)

metrics_path <- file.path(dirname(rds_input_dir), 'metrics.csv')
if (!file.exists(metrics_path)) {
  stop(paste0('Metrics file not found at ', metrics_path))
}
roi_metrics <- read.csv(metrics_path)
print(paste('Loaded ROI metrics from', metrics_path))
roi_df <- bind_metrics(gp.spr, roi_metrics)


# Determine right model to fit
rds_files <- list.files(
  rds_input_dir,
  pattern = '\\.[Rr][Dd][Ss]$',
  full.names = TRUE,
  recursive = FALSE
)
rds_files <- sort(rds_files)
print(paste('Found', length(rds_files), 'RDS files.'))


if (model_idx < 1 || model_idx > length(rds_files)) {
  stop(paste0('model_idx must be between 1 and ', length(rds_files), '; got ', model_idx))
}

selected_rds <- rds_files[[model_idx]]


prior <- c(
  prior('normal(900,2000)', class = 'Intercept'),
  prior('normal(0,450)', class = 'b'),
  prior('normal(0,300)', class = 'sd'),
  prior('normal(0,600)', class = 'sigma')
)

for (rds_file in selected_rds) {
  model_stem <- basename(rds_file)
  model_stem <- str_remove(model_stem, '\\.[Rr][Dd][Ss]$')
  print(paste('Processing', model_stem))

  model_name <- paste0('eachword_', model_stem)
  predicted <- Predicting_RT_with_spillover(roi_df, rds_file)

  for (roi in c(0)) {
    print(paste('Fitting brm for', model_name, 'ROI', roi, 'over', metrics_path))
    fit_data <- subset(predicted, ROI == roi & !is.na(predicted))

    model_fit <- brm(
      predicted ~ AMBUAMB * (SZM1 + SZM2) +
        (1 + AMBUAMB * (SZM1 + SZM2) || item) +
        + (1 | participant),
      data = fit_data,
      iter = niter,
      warmup = nwarmup,
      cores = ncores,
      chains = ncores,
      threads = threading(num_threads),
      backend = 'cmdstanr',
      control = list(adapt_delta = delta, max_treedepth = maxtree),
      prior = prior,
      silent = FALSE
    )

    out_path <- paste0(output_dir, '/', model_name, '.rds')
    saveRDS(model_fit, file = out_path)
    print(paste('Saved', out_path))

    rm(model_fit, fit_data, predicted)
    gc()
  }
}


print(paste0('Finished fitting ', selected_rds, ' brm model(s).'))
