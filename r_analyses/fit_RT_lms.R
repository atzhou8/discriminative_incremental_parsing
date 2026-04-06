library(ggplot2)
library(plyr)
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(stringr)
source('./util.R')

# Write a particular predictor of interest as surprisal and surprisal_s for 
# downstream code 
bind_metrics_parser <- function(spr, surps, predictor_col = "kl_forward") {
  nuisance_lookup <- surps_gpt2 %>%
    select(Sentence, word_pos, length_s, logfreq_s, gpt2_surprisal=sum_surprisal_s) %>%
    distinct()
  surps <- surps %>%
    left_join(nuisance_lookup, by = c("Sentence", "word_pos"), suffix = c("", ".lookup"))
  if (!("length_s" %in% names(surps)) && ("length_s.lookup" %in% names(surps))) {
    surps$length_s <- surps$length_s.lookup
  }
  if (!("logfreq_s" %in% names(surps)) && ("logfreq_s.lookup" %in% names(surps))) {
    surps$logfreq_s <- surps$logfreq_s.lookup
  }
  surps <- surps %>% select(-any_of(c("length_s.lookup", "logfreq_s.lookup")))

  if (!("length_s" %in% names(surps)) || !("logfreq_s" %in% names(surps))) {
    stop("Custom metrics data still missing 'length_s' and/or 'logfreq_s' after lookup merge.")
  }

  merged <- merge(x=spr, y=surps,
                  by.x=c("Sentence", "WordPosition"), by.y=c("Sentence", "word_pos"), 
                  all.x=TRUE)

  merged$item <- merged$item.x
  if (!(predictor_col %in% names(merged))) {
    stop(paste0("Predictor column not found: ", predictor_col))
  }
  predictor_vals <- suppressWarnings(as.numeric(merged[[predictor_col]]))
  merged$surprisal <- predictor_vals
  merged$surprisal_s <- suppressWarnings(as.numeric(merged[[paste0(predictor_col, '_s')]]))
  
  with_lags <- merged %>% group_by_at(vars(item, participant)) %>%
                    mutate(RT_p1 = lag(RT), 
                           RT_p2 = lag(RT_p1), 
                           RT_p3 = lag(RT_p2),
                           length_p1_s = lag(length_s), 
                           length_p2_s = lag(length_p1_s),
                           length_p3_s = lag(length_p2_s),
                           logfreq_p1_s = lag(logfreq_s), 
                           logfreq_p2_s = lag(logfreq_p1_s),
                           logfreq_p3_s = lag(logfreq_p2_s),
                           surprisal_p1_s = lag(surprisal_s),
                           surprisal_p2_s = lag(surprisal_p1_s),
                           surprisal_p3_s = lag(surprisal_p2_s)
                  )

  with_lags$sent_length <- lapply(str_split(with_lags$Sentence, " "), length)

  dropped <- subset(with_lags, !is.na(surprisal_s) &
                      !is.na(surprisal_p1_s) & 
                      !is.na(surprisal_p2_s) &
                      !is.na(surprisal_p3_s) &
                      !is.na(logfreq_s) & !is.na(logfreq_p1_s) &
                      !is.na(logfreq_p2_s) & !is.na(logfreq_p3_s) & 
                      (with_lags$sent_length != with_lags$WordPosition))

  print(paste0("dropped: ", nrow(with_lags) - nrow(dropped)))
  return(dropped)
}

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

surps_parser <- read.csv('../out/sud_with_adjunct/items_filler.parser.csv.scaled')
surps_lstm <- read.csv('./predictors/items_filler.lstm.csv.scaled')
surps_gpt2 <- read.csv('./predictors/items_filler.gpt2.csv.scaled')
spr <- load_data('Fillers')
for (metric in metrics_to_fit) {
  # Ensure nuisance predictors expected by the spillover model exist.
  print("Collating data...")
  dropped.parser <- bind_metrics_parser(spr, surps_parser, predictor_col = metric)
  print("Data collated.")

  print("Fitting spillover model...")
  models.filler.spillover <- lmer(
    RT ~ surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s +
      scale(WordPosition) + logfreq_s * length_s + logfreq_p1_s * length_p1_s +
      logfreq_p2_s * length_p2_s + logfreq_p3_s * length_p3_s +
      (1 + surprisal_s + surprisal_p1_s + surprisal_p2_s + surprisal_p3_s || participant) + (1 | item),
    data = dropped.parser
  )
  saveRDS(models.filler.spillover, paste0("filler_models/filler_parser_", metric, "_spillover.rds"))
  print("Spillover model fitted.")
  
  print("Fitting single word model...")
  models.filler.singleword <- lmer(
    RT ~ surprisal_s + scale(WordPosition) + logfreq_s * length_s + logfreq_p1_s * length_p1_s +
      logfreq_p2_s * length_p2_s + logfreq_p3_s * length_p3_s + (1 + surprisal_s || participant) + (1 | item),
    data = dropped.parser,
  )
  saveRDS(models.filler.singleword, paste0("filler_models/filler_parser_", metric, "_singleword.rds"))
  print("Single word model fitted.")

  print("Fitting one-lagged model...")
  models.filler.singleword <- lmer(
    RT ~ surprisal_p1_s + scale(WordPosition) + logfreq_s * length_s + logfreq_p1_s * length_p1_s +
      logfreq_p2_s * length_p2_s + logfreq_p3_s * length_p3_s + (1 + surprisal_p1_s || participant) + (1 | item),
    data = dropped.parser,
  )
  saveRDS(models.filler.singleword, paste0("filler_models/filler_parser_", metric, "_onelagged.rds"))
  print("One-lagged model fitted.")

  print("Fitting shared model...")
  models.filler.shared <- lmer(
    RT ~ surprisal_s + gpt2_surprisal + scale(WordPosition) + logfreq_s * length_s + logfreq_p1_s * length_p1_s +
      logfreq_p2_s * length_p2_s + logfreq_p3_s * length_p3_s + (1 + surprisal_s || participant) + (1 | item),
    data = dropped.parser,
  )
  saveRDS(models.filler.shared, paste0("filler_models/filler_parser_", metric, "_shared.rds"))
  print("Single word model fitted.")

  # rm(models.filler.spillover, models.filler.singleword, dropped.parser) # free memory
  gc()
}