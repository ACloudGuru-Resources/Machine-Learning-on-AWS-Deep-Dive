import pandas as pd
import datetime

timestamp = pd.to_datetime("now").timestamp()

df.insert(1,'event_time', timestamp)