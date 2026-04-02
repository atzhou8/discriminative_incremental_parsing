library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(stringr)
library(lme4)
source('./util.R')

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
# Collect results in an R data.frame (one row per model x ROI)
results <- data.frame(
  model = character(),
  ROI = integer(),
  NPS = numeric(),
  NPZ = numeric(),
  MVRR = numeric(),
  IsSingular = logical(),
  DeltaLogLikelihood = numeric(),
  p_metric = numeric(),
  p_gpt = numeric(),
  stringsAsFactors = FALSE,
  check.names = FALSE
)

for (metric in metrics_to_fit) {
  onelagged <- paste0("parser_", metric, "_shared")
  PredictedRT_df <- Predicting_RT_with_spillover(
    rt.data,
    "ClassicGP",
    models = c(onelagged),
    parser_predictor_col = metric
  )
  filler_model <- readRDS(paste0('./filler_models/filler_', onelagged, '.rds')) 
  summary(filler_model)
  for (roi in c(0, 1, 2)) {
    model_fit <- lmer(predicted ~ AMBUAMB*(SZM1+SZM2) +
                        (1 + AMBUAMB*(SZM1+SZM2) || item) +
                        (1 | participant),
                      data = subset(PredictedRT_df, model == onelagged & ROI == roi & !is.na(RT)))

    nps <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM1"]
    npz <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM2"]
    mvrr <- fixef(model_fit)["AMBUAMB"]
    results <- rbind(results, data.frame(
      model = onelagged,
      ROI = roi,
      NPS = nps,
      NPZ = npz,
      MVRR = mvrr,
      IsSingular = isSingular(filler_model),
      DeltaLogLikelihood = as.numeric(logLik(filler_model)) + 6577546 ,
      p_metric = coef(summary(filler_model))['surprisal_p1_s', 'Pr(>|t|)'],
      p_gpt = coef(summary(filler_model))['gpt2_surprisal', 'Pr(>|t|)'],
      stringsAsFactors = FALSE
    ))
  # single <- paste0("parser_", metric, "_singleword")
  # PredictedRT_df <- Predicting_RT_with_spillover(
  #   rt.data,
  #   "ClassicGP",
  #   models = c(single),
  #   parser_predictor_col = metric
  # )
  # filler_model <- readRDS(paste0('./filler_models/filler_', single, '.rds')) 
  # for (roi in c(0, 1, 2)) {
  #   model_fit <- lmer(predicted ~ AMBUAMB*(SZM1+SZM2) +
  #                       (1 + AMBUAMB*(SZM1+SZM2) || item) +
  #                       (1 | participant),
  #                     data = subset(PredictedRT_df, model == single & ROI == roi & !is.na(RT)))

  #   nps <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM1"]
  #   npz <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM2"]
  #   mvrr <- fixef(model_fit)["AMBUAMB"]
  #   results <- rbind(results, data.frame(
  #     model = single,
  #     ROI = roi,
  #     NPS = nps,
  #     NPZ = npz,
  #     MVRR = mvrr,
  #     IsSingular = isSingular(filler_model),
  #     DeltaLogLikelihood = as.numeric(logLik(filler_model)) + 6577546,
  #     p_metric = coef(summary(filler_model))['surprisal_s', 'Pr(>|t|)'],
  #     p_metric_p1 = NaN,
  #     p_metric_p2 = NaN,
  #     stringsAsFactors = FALSE
  #   ))
  # }

  # spillover <- paste0("parser_", metric, "_spillover")
  # PredictedRT_df <- Predicting_RT_with_spillover(
  #   rt.data,
  #   "ClassicGP",
  #   models = c(spillover),
  #   parser_predictor_col = metric
  # )
  # filler_model <- readRDS(paste0('./filler_models/filler_', spillover, '.rds')) 
  # for (roi in c(0, 1, 2)) {
  #   model_fit <- lmer(predicted ~ AMBUAMB*(SZM1+SZM2) +
  #                       (1 + AMBUAMB*(SZM1+SZM2) || item) +
  #                       (1 | participant),
  #                     data = subset(PredictedRT_df, model == spillover & ROI == roi & !is.na(RT)))

  #   nps <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM1"]
  #   npz <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM2"]
  #   mvrr <- fixef(model_fit)["AMBUAMB"]

  #   results <- rbind(results, data.frame(
  #     model = spillover,
  #     ROI = roi,
  #     NPS = nps,
  #     NPZ = npz,
  #     MVRR = mvrr,
  #     IsSingular = isSingular(filler_model),
  #     DeltaLogLikelihood = as.numeric(logLik(filler_model)) + 6577546,
  #     p_metric = coef(summary(filler_model))['surprisal_s', 'Pr(>|t|)'],
  #     p_metric_p1 = coef(summary(filler_model))['surprisal_p1_s', 'Pr(>|t|)'],
  #     p_metric_p2 = coef(summary(filler_model))['surprisal_p2_s', 'Pr(>|t|)'],
  #     stringsAsFactors = FALSE
  #   ))
  # }

  # onelagged <- paste0("parser_", metric, "_onelagged")
  # PredictedRT_df <- Predicting_RT_with_spillover(
  #   rt.data,
  #   "ClassicGP",
  #   models = c(onelagged),
  #   parser_predictor_col = metric
  # )
  # filler_model <- readRDS(paste0('./filler_models/filler_', onelagged, '.rds')) 
  # for (roi in c(0, 1, 2)) {
  #   model_fit <- lmer(predicted ~ AMBUAMB*(SZM1+SZM2) +
  #                       (1 + AMBUAMB*(SZM1+SZM2) || item) +
  #                       (1 | participant),
  #                     data = subset(PredictedRT_df, model == onelagged & ROI == roi & !is.na(RT)))

  #   nps <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM1"]
  #   npz <- fixef(model_fit)["AMBUAMB"] + fixef(model_fit)["AMBUAMB:SZM2"]
  #   mvrr <- fixef(model_fit)["AMBUAMB"]

  #   results <- rbind(results, data.frame(
  #     model = onelagged,
  #     ROI = roi,
  #     NPS = nps,
  #     NPZ = npz,
  #     MVRR = mvrr,
  #     IsSingular = isSingular(filler_model),
  #     DeltaLogLikelihood = as.numeric(logLik(filler_model)) + 6577546 ,
  #     p_metric = NaN,
  #     p_metric_p1 = coef(summary(filler_model))['surprisal_p1_s', 'Pr(>|t|)'],
  #     p_metric_p2 = NaN,
  #     stringsAsFactors = FALSE
  #   ))
  }
}

write.csv(results, file = 'EOIs_2.csv', row.names = FALSE)