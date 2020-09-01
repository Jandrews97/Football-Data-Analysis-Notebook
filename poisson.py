"""Poisson distribution for number of goals in a match"""

import os
import glob
import numpy as np
import pandas as pd

path = r"C:\Users\Jamie\OneDrive\Football Data\Football-Data.co.uk\Big 5 Leagues (05-06 to 18-19)"
EPL = glob.glob(os.path.join(path, "E0*.csv"))
df = pd.concat(pd.read_csv(f) for f in EPL)

df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "B365H", "B365D", "B365A"]]
