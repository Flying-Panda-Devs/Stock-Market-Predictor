from alive_progress import alive_bar
print('Importing modules...')
with alive_bar(8) as bar:
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
    import os
    bar()
    import argparse
    bar()
def convertDateToInt(date:str):
    return datetime.strptime(date,'%Y-%m-%d').timestamp()
earliest = convertDateToInt('2013-02-08')
latest = convertDateToInt('2018-02-07')
parser = argparse.ArgumentParser(
    prog='stock-predictor'
)
parser.add_argument('-r','--reset',action='store_true')
parser.add_argument('epochs')
args = parser.parse_args()

if args.reset:
    print('Reset flag detected, deleting all cached models...')
    with alive_bar(len(os.listdir('working_data/trained_models/'))) as bar:
        for file in os.listdir('working_data/trained_models/'):
            bar()
            if '.pt' in file:
                os.remove(f'working_data/trained_models/{file}')
            
torch.set_printoptions(precision=15,sci_mode=False)
all_stock_data = pd.read_csv('static_data/all_stocks_5yr.csv')
stock_names = list(set(all_stock_data['Name'].tolist()))
per_stock_data = dict()
print('Loading and transforming static data...')
with alive_bar(len(stock_names)) as bar:
    for stock_name in stock_names:
        per_stock_data[stock_name] = all_stock_data[all_stock_data['Name'] == stock_name].drop(columns=['Name','volume'])
        per_stock_data[stock_name]['date'] = per_stock_data[stock_name]['date'].apply(np.vectorize(convertDateToInt))
        per_stock_data[stock_name] = per_stock_data[stock_name].astype(np.float32)
        bar()
tensors = dict()
tensors['input'] = dict()
tensors['output'] = dict()
print('Generating tensors...')
with alive_bar(len(stock_names)) as bar:
    for stock_name in stock_names: # some of the tensors are weirdly small. check stock data.
        tensors['input'][stock_name] = torch.from_numpy(per_stock_data[stock_name].drop(per_stock_data[stock_name].tail(1).index).drop(columns=['date']).to_numpy())
        tensors['output'][stock_name] = torch.from_numpy(per_stock_data[stock_name].drop(per_stock_data[stock_name].head(1).index).drop(columns=['date']).to_numpy())
        bar()
print('Loading cached models...')
models = dict()
with alive_bar(len(stock_names)) as bar:
    for stock_name in stock_names:
        if os.path.exists(f'working_data/trained_models/{stock_name}.pt'):
            model = nn.Sequential(
                nn.Linear(4, 500),
                nn.BatchNorm1d(500),
                nn.LeakyReLU(),
                nn.Linear(500,120),
                nn.BatchNorm1d(120),
                nn.LeakyReLU(),
                nn.Linear(120,12),
                nn.BatchNorm1d(12),
                nn.Linear(12,4),
                nn.ReLU()
            )
            model.load_state_dict(torch.load(f'working_data/trained_models/{stock_name}.pt', weights_only = True))
            models[stock_name] = model
        bar()
print('All cached models loaded.')
stock_name = input('Stock name: ')
stock_date = convertDateToInt(input('Target date (YYYY-MM-DD): '))
print('Loading model...')
if not (stock_name in list(models.keys())):
    print('Model not in cache.')
    # Define the model using nn.Sequential
    model = nn.Sequential(
        nn.Linear(4, 500),
        nn.BatchNorm1d(500),
        nn.LeakyReLU(),
        nn.Linear(500,120),
        nn.BatchNorm1d(120),
        nn.LeakyReLU(),
        nn.Linear(120,12),
        nn.BatchNorm1d(12),
        nn.Linear(12,4),
        nn.ReLU()
    )

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)  

    # Train the model for 500 epochs
    print('Training new model...')
    with alive_bar(int(args.epochs)) as bar:
        for epoch in range(int(args.epochs)):  
            model.train()  # Set the model to training mode
            optimizer.zero_grad()  # Zero the gradients for iteration
            outputs = model(tensors['input'][stock_name])  # Compute predictions
            loss = criterion(outputs, tensors['output'][stock_name])  # Compute the loss
            loss.backward()  # Compute the gradient of the loss
            optimizer.step()  # Optimize the model parameters
            bar()
    models[stock_name] = model
    print('Caching model...')
    torch.save(model.state_dict(),f'working_data/trained_models/{stock_name}.pt')
else:
    print('Model in cache.')
active_model = models[stock_name]
active_model.eval()
print('Model loaded. Ready for evaluation.')
with torch.no_grad():
    if earliest < stock_date < latest:
        new_input = torch.from_numpy(per_stock_data[stock_name].loc[per_stock_data[stock_name]['date'] == stock_date].to_numpy())
        diff = 0
    else:
        new_input = torch.from_numpy(per_stock_data[stock_name].loc[per_stock_data[stock_name]['date'] == latest].to_numpy())
        diff = datetime.fromtimestamp(stock_date) - datetime.fromtimestamp(latest)
        diff = diff.days
    if diff != 0:
        for i in range(diff):
            prediction = active_model(new_input)
            new_input = prediction
    else:
        prediction = active_model(new_input)
    print('Evaluation complete. Result:')
    print(prediction)