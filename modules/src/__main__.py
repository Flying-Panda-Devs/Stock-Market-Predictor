from alive_progress import alive_bar
with alive_bar(6) as bar:
    import torch
    bar()
    import pandas as pd
    bar()
    import numpy as np
    bar()
    from datetime import datetime
    bar()
    import torch.nn as nn
    bar()
    import torch.optim as optim
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
        per_stock_data[stock_name] = per_stock_data[stock_name].astype(np.float32)
        bar()
tensors = dict()
tensors['input'] = dict()
tensors['output'] = dict()
with alive_bar(len(stock_names)) as bar:
    for stock_name in stock_names: # some of the tensors are weirdly small. check stock data.
        tensors['input'][stock_name] = torch.from_numpy(per_stock_data[stock_name].drop(per_stock_data[stock_name].tail(1).index).to_numpy())
        tensors['output'][stock_name] = torch.from_numpy(per_stock_data[stock_name].drop(per_stock_data[stock_name].head(1).index).to_numpy())
        print(tensors['input'][stock_name].shape,tensors['output'][stock_name].shape)
        bar()
models = dict()
with alive_bar(len(stock_names)) as bar:
    for stock_name in stock_names:
        # Define the model using nn.Sequential
        model = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

        # Define loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)  

        # Train the model for 50 epochs
        for epoch in range(50):  
            model.train()  # Set the model to training mode
            optimizer.zero_grad()  # Zero the gradients for iteration
            outputs = model(tensors['input'][stock_name])  # Compute predictions
            loss = criterion(outputs, tensors['output'][stock_name])  # Compute the loss
            loss.backward()  # Compute the gradient of the loss
            optimizer.step()  # Optimize the model parameters
        models[stock_name] = model
