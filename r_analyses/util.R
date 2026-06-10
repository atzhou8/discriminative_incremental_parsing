library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(posterior)
library(tidybayes)
library(tidyverse)

load_data <- function(subsetname,RTcutoffhigh=7000,RTcutofflow=0){
  
  #id <- ifelse(subsetname=="ClassicGP","1TanAUeI1x_G0mkFsFnpru0udi8oRvxNH",
  #              ifelse(subsetname=="RelativeClause","1ndHGJxTV51AEpJ2MKQxpHUpmCuj-Lm-W",
  #                ifelse(subsetname=="AttachmentAmbiguity","1TShRMEgba4z0tgN5zj48-k-3FB-hl5Gj",
  #                  ifelse(subsetname=="Agreement","1V6m9d20CbB1GeadR6SQ6yeiRyoqz3zwJ",
  #                         ifelse(subsetname=="Fillers","16onEVQVsFgusBZXuVPrtwcIiDaJgV6WB","")))))
  #rt.data <- read.csv(sprintf("https://docs.google.com/uc?id=%s&export=download&confirm=t", id), header=TRUE) %>% mutate(participant=MD5)
  #please note currently large files on google drive can't be loaded with url (due to the virus scanning warning)
  #please download the file manually first to the local folder
  
  id <- ifelse(subsetname=="ClassicGP","items/ClassicGardenPathSet_merged.csv",
                ifelse(subsetname=="RelativeClause","items/RelativeClauseSet_merged.csv",
                  ifelse(subsetname=="AttachmentAmbiguity","items/AttachmentSet_merged.csv",
                    ifelse(subsetname=="Agreement","items/AgreementSet_merged.csv",
                           ifelse(subsetname=="Fillers","items/Fillers_merged.csv","")))))
  rt.data <- read.csv(id, header=TRUE) %>% mutate(participant=MD5)
  rt.data$RT <- ifelse(rt.data$RT>RTcutoffhigh,NA,ifelse(rt.data$RT<RTcutofflow,NA,rt.data$RT))
  rt.data$Sentence <- str_replace_all(rt.data$Sentence, "%2C", ",")
  rt.data$EachWord <- str_replace_all(rt.data$EachWord, "%2C", ",")
  rt.data$word <-  tolower(ifelse(substring(rt.data$EachWord,nchar(rt.data$EachWord),nchar(rt.data$EachWord))%in%c(".",","),substring(rt.data$EachWord,1,nchar(rt.data$EachWord)-1),rt.data$EachWord))
  return(rt.data)
}

Predicting_RT_with_spillover <- function(rt.data, model_dir){
  filler.model <- readRDS(model_dir) 
  rt.data <- subset(
    rt.data,
    ROI == 0
  )
  rt.data$predicted <- predict(filler.model, newdata=rt.data, allow.new.levels = TRUE)
  
  return(rt.data)
}

bind_metrics <- function(spr, metrics) {
  merged <- merge(x=spr, y=metrics,
                  by.x=c("Sentence", "WordPosition"), by.y=c("Sentence", "word_pos"), 
                  all.x=TRUE)
  merged$item <- merged$item.x
  return(merged)
}

fit_model_data <- function(df, cols) {
  cols <- cols[cols %in% names(df)]
  df[, cols, drop = FALSE]
}

add_per_word_cols <- function(df, col_names, offsets = c(0, 1, 2)) {
  for (i in seq_along(offsets)) {
    offset <- offsets[[i]]
    pos_col <- paste0("word_pos_lookup_", i)

    # Build lookup with Sentence, word_pos, and selected columns
    lookup_cols <- c("Sentence", "word_pos", col_names)
    lookup <- df %>%
      select(all_of(lookup_cols)) %>%
      distinct()
    
    # Rename word_pos to position lookup column
    names(lookup)[names(lookup) == "word_pos"] <- pos_col
    
    # Rename each column to add the suffix _wi
    for (col in col_names) {
      if (col %in% names(lookup)) {
        new_name <- paste0(col, "_w", i)
        lookup <- lookup %>% rename(!!new_name := !!col)
      }
    }

    df[[pos_col]] <- df$word_pos + offset
    df <- merge(
      x = df,
      y = lookup,
      by.x = c("Sentence", pos_col),
      by.y = c("Sentence", pos_col),
      all.x = TRUE
    )
    df[[pos_col]] <- NULL
  }
  df
}

get_scale_params <- function(df, cols) {
  params <- list()
  for (col in cols) {
    values <- df[[col]]
    mean_val <- mean(values, na.rm=TRUE)
    sd_val <- sd(values, na.rm=TRUE)
    params[[col]] <- list(mean=mean_val, sd=sd_val)
  }
  params
}

apply_scale_params <- function(df, params) {
  for (col in names(params)) {
    stats <- params[[col]]
    df[[paste0(col, "_s")]] <- (df[[col]] - stats$mean) / stats$sd
  }
  df
}
