# -- Packages
"""
Pour récupérer les données, nous allons utiliser la librarie Python "yfinance"
Elle va contacter les serveurs du site Yahoo Finances, à travers l'API du site, pour en récupérer les données des cours choisis.
Nous avons plusieurs ticks disponibles : 1m, 15m, 30m, 60m, daily

Pour le projet, nous allons récupérer les données, avec un tick de 1j, des actions : 
- Total
- Sanofi
- BNP Paribas
- LVMH
- Michelin
La limitation à des indices journaliers et non intradays et dû au fait que nous contactons l'API sans compte Yahoo payant. 
Les ticks intradays ne peuvent être récupérés que sur 30 jours maximum. 
"""
# %% 
import yfinance as yf
import pandas as pd 
import matplotlib.pyplot as plt 
import os

# %% 
# -- Get data
# - Données complètes
dataset = yf.download(tickers = ["FP.PA", "SAN.PA", "BNP.PA", "MC.PA", "ML.PA"])
dataset

# %% 
# - Limitation au prix de clôture
dataset.columns # Format MultiIndex de Pandas
dataset = dataset["Close"]

# - Changer le noms des colonnes
dataset.rename(columns={"FP.PA":"Total", "SAN.PA":"Sanofi",
                         "BNP.PA" : "BNP", "MC.PA":"LVMH", "ML.PA":"Michelin"})

# %%
# - Grahpiques d'évolutions des cours
"""
Cela permet également de connaître à quel moment les cours démarrent.
"""
dataset.plot(figsize = (16,8))
plt.ylabel("Prix en €")

# %% 
# - Data processing
dataset = dataset.loc[dataset.index >= '2000-01-03T00:00:00.000']
dataset.dropna(inplace = True)
dataset = dataset.reset_index()

# %% 
# - Save dataset
path = "/Users/valentinjoly/Documents/GitHub/Reinforcement-Learning-Project---DS2E/Data"
dataset.loc[:, dataset.columns != 'Date'].to_csv(os.path.join(path, r"Dataset_full.csv"), index = False)