from alive_progress import alive_bar
with alive_bar(4) as bar:
    import torch
    bar()
    import pandas as pd
    bar()
    import numpy as np
    bar()
    from datetime import datetime
    bar()

def convertDateToInt(date:str):
    return datetime.strptime(date,'%Y-%m-%d').timestamp()

all_stock_data = pd.read_csv('data/all_stocks_5yr.csv')
stock_names = list(set(all_stock_data['Name'].tolist()))
per_stock_data = dict()
with alive_bar(len(stock_names)) as bar:
    for stock_name in stock_names:
        per_stock_data[stock_name] = all_stock_data[all_stock_data['Name'] == stock_name].drop(columns=['Name'])
        per_stock_data[stock_name]['date'] = per_stock_data[stock_name]['date'].apply(np.vectorize(convertDateToInt))
        bar()
tensors = dict()
tensors['input'] = dict()
tensors['output'] = dict()
with alive_bar(len(stock_names)) as bar:
    for stock_name in stock_names:
        tensors['input'][stock_name] = torch.from_numpy(per_stock_data[stock_name].drop(df.tail(1).index,inplace=True))
        tensors['output'][stock_name] = torch.from_numpy(per_stock_data[stock_name].drop(df.head(1).index,inplace=True))
        bar()
