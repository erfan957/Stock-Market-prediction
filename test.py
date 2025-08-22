import pandas as pd
import numpy as np
data = pd.read_csv("S_Saipa.csv")
data["<DTYYYYMMDD>"] = pd.to_datetime(data["<DTYYYYMMDD>"])
start_date = "2025-08-20"
end_date = "2025-03-18"
filtered_data = data[((data["<DTYYYYMMDD>"] >= start_date) & (data["<DTYYYYMMDD>"] <= end_date))]
print(filtered_data)