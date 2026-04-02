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
results <- data.frame(
  model = character(),
  R2 = numeric(),
  ROI = integer(),
  NPS = numeric(),
  NPZ = numeric(),
  MVRR = numeric(),
  NPS_CI_low = numeric(),
  NPS_CI_high = numeric(),
  NPZ_CI_low = numeric(),
  NPZ_CI_high = numeric(),
  MVRR_CI_low = numeric(),
  MVRR_CI_high = numeric(),
  DeltaLogLikelihood = numeric(),
  p_metric = numeric(),
  p_gpt = numeric(),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

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
    

  #   fe <- fixef(model_fit)[, "Estimate"]
  #   vc <- try(vcov(model_fit), silent = TRUE)
  #   z <- qnorm(0.975)
  #   nps_var <- vc["AMBUAMB", "AMBUAMB"] + vc["AMBUAMB:SZM1", "AMBUAMB:SZM1"] + 2 * vc["AMBUAMB", "AMBUAMB:SZM1"]
  #   npz_var <- vc["AMBUAMB", "AMBUAMB"] + vc["AMBUAMB:SZM2", "AMBUAMB:SZM2"] + 2 * vc["AMBUAMB", "AMBUAMB:SZM2"]
  #   mvrr_var <- vc["AMBUAMB", "AMBUAMB"]
    

  #   nps <- fe["AMBUAMB"] + fe["AMBUAMB:SZM1"]
  #   npz <- fe["AMBUAMB"] + fe["AMBUAMB:SZM2"]
  #   mvrr <- fe["AMBUAMB"]

  #   nps_ci_low <- nps - z * sqrt(nps_var)
  #   nps_ci_high <- nps + z * sqrt(nps_var)
  #   npz_ci_low <- npz - z * sqrt(npz_var)
  #   npz_ci_high <- npz + z * sqrt(npz_var)
  #   mvrr_ci_low <- mvrr - z * sqrt(mvrr_var)
  #   mvrr_ci_high <- mvrr + z * sqrt(mvrr_var)
  #   results <- rbind(results, data.frame(
  #     model = single,
  #     R2 = bayes_R2(model_fit),
  #     ROI = roi,
  #     NPS = nps,
  #     NPS_CI_low = nps_ci_low,
  #     NPS_CI_high = nps_ci_high,
  #     NPZ = npz,
  #     NPZ_CI_low = npz_ci_low,
  #     NPZ_CI_high = npz_ci_high,
  #     MVRR = mvrr,
  #     MVRR_CI_low = mvrr_ci_low,
  #     MVRR_CI_high = mvrr_ci_high,
  #     DeltaLogLikelihood = as.numeric(logLik(filler_model)) + 6577546,
  #     p_metric = coef(summary(filler_model))['surprisal_s', 'Pr(>|t|)'],
  #     p_metric_p1 = NaN,
  #     p_metric_p2 = NaN,
  #     stringsAsFactors = FALSE
  #   ))
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

    # fe <- fixef(model_fit)[, "Estimate"]
    # vc <- try(vcov(model_fit), silent = TRUE)
    # z <- qnorm(0.975)
    # nps_var <- vc["AMBUAMB", "AMBUAMB"] + vc["AMBUAMB:SZM1", "AMBUAMB:SZM1"] + 2 * vc["AMBUAMB", "AMBUAMB:SZM1"]
    # npz_var <- vc["AMBUAMB", "AMBUAMB"] + vc["AMBUAMB:SZM2", "AMBUAMB:SZM2"] + 2 * vc["AMBUAMB", "AMBUAMB:SZM2"]
    # mvrr_var <- vc["AMBUAMB", "AMBUAMB"]
    

    # nps <- fe["AMBUAMB"] + fe["AMBUAMB:SZM1"]
    # npz <- fe["AMBUAMB"] + fe["AMBUAMB:SZM2"]
    # mvrr <- fe["AMBUAMB"]

    # nps_ci_low <- nps - z * sqrt(nps_var)
    # nps_ci_high <- nps + z * sqrt(nps_var)
    # npz_ci_low <- npz - z * sqrt(npz_var)
    # npz_ci_high <- npz + z * sqrt(npz_var)
    # mvrr_ci_low <- mvrr - z * sqrt(mvrr_var)
    # mvrr_ci_high <- mvrr + z * sqrt(mvrr_var)

    # results <- rbind(results, data.frame(
    #   model = spillover,
    #   R2 = bayes_R2(model_fit),
    #   ROI = roi,
    #   NPS = nps,
    #   NPS_CI_low = nps_ci_low,
    #   NPS_CI_high = nps_ci_high,
    #   NPZ = npz,
    #   NPZ_CI_low = npz_ci_low,
    #   NPZ_CI_high = npz_ci_high,
    #   MVRR = mvrr,
    #   MVRR_CI_low = mvrr_ci_low,
    #   MVRR_CI_high = mvrr_ci_high,
    #   DeltaLogLikelihood = as.numeric(logLik(filler_model)) + 6577546,
    #   p_metric = coef(summary(filler_model))['surprisal_s', 'Pr(>|t|)'],
    #   p_metric_p1 = coef(summary(filler_model))['surprisal_p1_s', 'Pr(>|t|)'],
    #   p_metric_p2 = coef(summary(filler_model))['surprisal_p2_s', 'Pr(>|t|)'],
    #   stringsAsFactors = FALSE
    # ))
  }

}

write.csv(results, file = 'EOIs_brm.csv', row.names = FALSE)
