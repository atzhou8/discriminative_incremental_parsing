library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(stringr)
source('./util.R')

rt.data.gp <- load_data("ClassicGP")
rt.data.filler <- load_data("Fillers")
rt.data.gp$SZM1 <- ifelse(rt.data.gp$CONSTRUCTION=="NPS",1,0)
rt.data.gp$SZM2 <- ifelse(rt.data.gp$CONSTRUCTION=="NPZ",1,0)

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

for (metric in metrics_to_fit) {
  single <- paste0("parser_", metric, "_singleword")
  PredictedRT_df <- Predicting_RT_with_spillover(
    rt.data.filler,
    "filler",
    models = c(single),
    parser_predictor_col = metric
  )
  PredictedRT_df$List <- NULL
  PredictedRT_df$sent_length <- NULL
  write.csv(PredictedRT_df, file = paste0("RTs/filler_parser_", metric, "_singleword.csv"), row.names = FALSE)

  spillover <- paste0("parser_", metric, "_spillover")
  PredictedRT_df <- Predicting_RT_with_spillover(
    rt.data.filler,
    "filler",
    models = c(spillover),
    parser_predictor_col = metric
  )
  PredictedRT_df$List <- NULL
  PredictedRT_df$sent_length <- NULL
  write.csv(PredictedRT_df, file = paste0("RTs/filler_parser_", metric, "_spillover.csv"), row.names = FALSE)

  single <- paste0("parser_", metric, "_singleword")
  PredictedRT_df <- Predicting_RT_with_spillover(
    rt.data.gp,
    "ClassicGP",
    models = c(single),
    parser_predictor_col = metric
  )
  PredictedRT_df$List <- NULL
  PredictedRT_df$sent_length <- NULL
  write.csv(PredictedRT_df, file = paste0("RTs/gp_parser_", metric, "_singleword.csv"), row.names = FALSE)

  spillover <- paste0("parser_", metric, "_spillover")
  PredictedRT_df <- Predicting_RT_with_spillover(
    rt.data.gp,
    "ClassicGP",
    models = c(spillover),
    parser_predictor_col = metric
  )
  PredictedRT_df$List <- NULL
  PredictedRT_df$sent_length <- NULL
  write.csv(PredictedRT_df, file = paste0("RTs/gp_parser_", metric, "_spillover.csv"), row.names = FALSE)



}