library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(stringr)
source('./util.R')

metrics_to_fit <- c(
 'kl_backward',
 'renyi_divergence_backward_2',
 'renyi_divergence_backward_3',
 'renyi_divergence_backward_4',
 'renyi_divergence_backward_5',
 'renyi_divergence_backward_6',
 'cross_entropy_backward',
 'renyi_crossent_backward_2',
 'renyi_crossent_backward_3',
 'renyi_crossent_backward_4',
 'renyi_crossent_backward_5',
 'roberta_surp',
 'gpt2_surp',
 'synsurp',
 'ccg_kl'
)

cols_to_expand <- c(
  metrics_to_fit,
  'logfreq',
  'length'
)

# Load metrics and scale according to ROIs or full dataset
metrics_gp = read.csv('./predictors/all_predictors.ClassicGP_merged.csv')
metrics_gp <- add_per_word_cols(metrics_gp, col_names = cols_to_expand)
metrics_roi = subset(
  metrics_gp, word_pos == ifelse(ambiguity == 'ambiguous', disambPositionAmb, disambPositionUnamb)
)
print('Loaded merged predictors.')

cols_to_scale <- c(
  cols_to_expand, 
  'word_pos', 
  paste0(cols_to_expand, '_merged'),
  paste0(cols_to_expand, '_w1'),
  paste0(cols_to_expand, '_w2'),
  paste0(cols_to_expand, '_w3')
)
scale_params_roi <- get_scale_params(metrics_roi, cols_to_scale)
metrics_roi <- apply_scale_params(metrics_roi, scale_params_roi)
print('Applied ROI-only scaling to ROI dataset.')

write.csv(metrics_roi, 'results/rt_models/mergedRT/roi/metrics.csv', row.names=FALSE)
print('Wrote scaled ROI dataset.')


# Load RTs and bind with metrics
spr_gp <- load_data('ClassicGP')
print('Loaded RT datasets.')

# ROIs only
roi_data = bind_metrics(spr_gp, metrics_roi)
roi_data = subset(roi_data, ROI==0 & !is.na(RT_merged) & !is.na(Sentence))
print(paste('ROI data rows:', nrow(roi_data)))


# # Model fitting
lmer_ctrl <- lmerControl(optimizer = 'bobyqa', optCtrl = list(maxfun = 200000))

# ROIs only
for (metric in metrics_to_fit) {
  print(paste('Fitting each word ROI-only model for', metric))
  metric_w1_s <- paste0(metric, '_w1_s')
  metric_w2_s <- paste0(metric, '_w2_s')
  metric_w3_s <- paste0(metric, '_w3_s')
  formula <- paste0(
    'RT_merged ~ ', metric_w1_s, ' + ', metric_w2_s, ' + ', metric_w3_s,
    ' + logfreq_w1_s * length_w1_s + logfreq_w2_s * length_w2_s',
    ' + logfreq_w3_s * length_w3_s',
    ' + (1 + ', metric_w1_s, ' + ', metric_w2_s, ' + ', metric_w3_s, ' || participant) + (1 | item)'
  )
  model <- lmer(
    as.formula(formula),
    data=roi_data,
    REML = FALSE,
    control = lmer_ctrl
  )
  saveRDS(model, paste0('results/rt_models/mergedRT/roi/eachword/', metric, '.RDS'))
} 
