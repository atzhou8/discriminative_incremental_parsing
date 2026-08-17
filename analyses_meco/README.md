Documenting analysis pipeline here

- Get passages with ../data/phenomena/MECO/my_data/make_passages.py

- Get English-only measures with ../meco.ipynb

- Get info metrics into `../metrics/MECO/{lang}/{model_name}/{passage}.csv' using `compute_metrics_for_sentences.py'

- Append log frequency and word length from COCA using ./add_baselines.py

-  Fit lmers to reading time using ./fit_lmer.py
