library(lme4)
library(tidyr)
library(stringr)
library(posterior)
library(tidybayes)
library(tidyverse)
library(dplyr)
library(brms)

emp_P0 <- readRDS('eoi_models/empirical_prior1_fit_ROI0.rds')
emp_P1 <- readRDS('eoi_models/empirical_prior1_fit_ROI1.rds')
emp_P2 <- readRDS('eoi_models/empirical_prior1_fit_ROI2.rds')

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

posterior_samp <- posterior_samples(emp_P0)
randomslope_names <- colnames(posterior_samp)[grepl('r_item.+(AMBUAMB|SZM|Intercept)',colnames(posterior_samp))]

df = {
    'metric': [],
    'ROI': [],
    
}

for (metric in metrics_to_fit) {
    single <- paste0("parser_", metric, "_singleword")
    for (roi in c(0)) {
        model_fit = readRDS(paste0('./eoi_models/', single, '_ROI', roi, '.rds'))

        emp_samples <- posterior_samples(
            emp_P0,
            fixed=TRUE,
            pars=c(
                'b_Intercept',
                'b_AMBUAMB',
                'b_SZM1',
                'b_SZM2',
                'b_AMBUAMB:SZM1',
                'b_AMBUAMB:SZM2',
                randomslope_names
            )
        )
        predicted_samples <- posterior_samples(
            model_fit,
            fixed=TRUE,
            pars=c(
                'b_Intercept',
                'b_AMBUAMB',
                'b_SZM1',
                'b_SZM2',
                'b_AMBUAMB:SZM1',
                'b_AMBUAMB:SZM2',
                randomslope_names
            )
        )
    }

}