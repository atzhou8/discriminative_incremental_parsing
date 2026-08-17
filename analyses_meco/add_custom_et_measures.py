import re

import numpy as np
import pandas as pd

from pathlib import Path

et_dir = Path('../data/phenomena/MECO/my_data/english_measures.csv')
et_df = pd.read_csv(et_dir)

regressive_gopast = []

for i, row in et_df.iterrows():
    go_past = float(row['firstrun.gopast'])
    go_past_sel = float(row['firstrun.gopast.sel'])
    regressive_gopast.append(go_past - go_past_sel)


et_df['regressive.gopast'] = regressive_gopast
et_df.to_csv(et_dir)