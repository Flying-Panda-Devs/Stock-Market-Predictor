import torch
import pandas as pd
from datetime import date

stock_data = pd.read_csv('data/all_stocks_5yr.csv')
stock_data['date'] = stock_data['date'].astype("datetime.date")
print(stock_data.dtypes)
stock_tensor = torch.from_numpy(stock_data.values)
print(stock_tensor)
print(stock_tensor.shape)