library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(brms)
library(stringr)
source('./util.R')


args <- commandArgs(trailingOnly = TRUE)
metric_idx <- if (length(args) >= 1 && nzchar(args[1])) as.integer(args[1]) else NA_integer_

rt.data <- load_data("ClassicGP")
rt.data$SZM1 <- ifelse(rt.data$CONSTRUCTION=="NPS",1,0)
rt.data$SZM2 <- ifelse(rt.data$CONSTRUCTION=="NPZ",1,0)

metrics_to_fit <- c(
  'kl_forward',
  'kl_backward',
  'kl_forward_recovered',
  'kl_backward_recovered',
  'js_geo',
  'js_geo_recovered',
  'renyi_divergence_forward_5',
  'renyi_divergence_forward_5_recovered',
  'renyi_divergence_backward_5',
  'renyi_divergence_backward_5_recovered'
)

if (!is.na(metric_idx)) {
  if (metric_idx < 1 || metric_idx > length(metrics_to_fit)) {
    stop(paste0("metric_idx must be between 1 and ", length(metrics_to_fit), "; got ", metric_idx))
  }
  metrics_to_fit <- metrics_to_fit[metric_idx]
}

# Collect results in an R data.frame (one row per model x ROI)
get_brms_parameters <- function(prior_type){
  
  if(prior_type == 'prior1'){
    curr_prior = c(prior("normal(300,1000)", class = "Intercept"),
                   prior("normal(0,150)", class = "b"),  
                   prior("normal(0,200)", class = "sd"),    
                   prior("normal(0,500)", class = "sigma"))
  }else if(prior_type == 'prior2'){
    curr_prior = c(prior("normal(300,1000)", class = "Intercept"),
                   prior("normal(0,100)", class = "b"),  
                   prior("normal(0,200)", class = "sd"),
                   prior("normal(0,500)", class = "sigma"))
  }else if(prior_type == 'prior3'){
    curr_prior =c(prior("normal(300,1000)", class = "Intercept"),
                  prior("normal(0,100)", class = "b"),  
                  prior("normal(0,150)", class = "sd"),
                  prior("normal(0,300)", class = "sigma"))
  }else if(prior_type == 'prior_bernoulli'){
    curr_prior = c(prior(normal(-1.5,1),class='Intercept')
                   ,prior(normal(0, 0.75),class = 'b'))
  }
  else{
    print('ENTER A VALID PRIOR')
  }
  
  parms <- list(prior = curr_prior,
                ncores = 4,
                niters = 12000,
                seed = 117,
                warmup = 6000,
                adapt_delta = 0.8)
  
  return(parms)
}

brm_param_list <- get_brms_parameters("prior1")
prior1 <- c(prior("normal(300,1000)", class = "Intercept"),
            prior("normal(0,150)", class = "b"),  
            prior("normal(0,200)", class = "sd"),    
            prior("normal(0,500)", class = "sigma"))
for (metric in metrics_to_fit) {
  single <- paste0("parser_", metric, "_singleword")
  PredictedRT_df <- Predicting_RT_with_spillover(
    rt.data,
    "ClassicGP",
    models = c(single),
    parser_predictor_col = metric
  )
  filler_model <- readRDS(paste0('./filler_models/filler_', single, '.rds')) 
  for (roi in c(0)) {
    model_fit <- brm(
      predicted ~ AMBUAMB*(SZM1+SZM2) +
                  (1 + AMBUAMB*(SZM1+SZM2) || item) +
                  (1 | participant),
      data = subset(PredictedRT_df, model == single & ROI == roi & !is.na(RT)),
      iter=24000,
      cores=4,
      warmup=12000,
      seed=brm_param_list$seed,
      prior=prior1,
      control=list(adapt_delta=brm_param_list$adapt_delta),
      silent=FALSE,
    )
    saveRDS(model_fit, file=paste0('./eoi_models/', single, '_ROI', roi, '.rds'))
    rm(model_fit)
    gc()
  }

  spillover <- paste0("parser_", metric, "_spillover")
  PredictedRT_df <- Predicting_RT_with_spillover(
    rt.data,
    "ClassicGP",
    models = c(spillover),
    parser_predictor_col = metric
  )
  filler_model <- readRDS(paste0('./filler_models/filler_', spillover, '.rds')) 
  for (roi in c(0)) {
    model_fit <- brm(
      predicted ~ AMBUAMB*(SZM1+SZM2) +
                  (1 + AMBUAMB*(SZM1+SZM2) || item) +
                  (1 | participant),
      data = subset(PredictedRT_df, model == spillover & ROI == roi & !is.na(RT)),
      iter=24000,
      cores=4,
      warmup=12000,
      seed=brm_param_list$seed,
      prior=prior1,
      control=list(adapt_delta=brm_param_list$adapt_delta)
    )
    saveRDS(model_fit, file=paste0('./eoi_models/', spillover, '_ROI', roi, '.rds'))
    rm(model_fit)
    gc()
  }

}

write.csv(results, file = 'EOIs_brm.csv', row.names = FALSE)
