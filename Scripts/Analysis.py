"""
RL Prject : Analyse
"""

# --- 1. Chargement des librairies nécessaires et données
# %% 
# Libraries
import pandas as pd
import numpy as np 

# %%
# --- 2. Definition des strategies financières de bases
class Agent:
    def __init__(self, capital, commission):
        self.capital = capital #Correspond à l'argent que l'agent peut investir
        self.commission = commission #Commission pour le passage d'un ordre
        self.stocks = pd.read_csv("/Users/valentinjoly/Documents/GitHub/Reinforcement-Learning-Project---DS2E/Data/Dataset_full.csv") #Représente le dataset 

    def market_strategy(self):
        """
        ...
        """
        # Définition des variables
        capital = self.capital
        
        stocks = self.stocks.copy()
        stocks.dropna(inplace = True)
        stocks.reset_index()

        pf_actions = pd.DataFrame(np.zeros((self.stocks.shape[0], 4)),
                                  columns = ["V_bourse", "V_portefeuille", "Gains", "Cum_gains"])

        # Definition des variables en t = 0
        pf_actions["V_bourse"][0] = round(stocks.loc[0, 'BNP.PA':'SAN.PA'].sum(), 2)

        nb_actions = int(capital / pf_actions["V_bourse"][0])
        pf_actions["V_portefeuille"][0] = round(nb_actions * pf_actions["V_bourse"][0], 2)
        pf_actions["Gains"][0] = 0
        pf_actions["Cum_gains"][0] = 0

        capital = capital - pf_actions["V_portefeuille"][0] # ce qu'il reste après investissement

        # Execution
        for i in range(1, len(stocks)):
            # Mise à jour du portefeuille d'actions
            pf_actions["V_bourse"][i] = round(stocks.loc[i, 'BNP.PA':'SAN.PA'].sum(), 2)
            pf_actions["V_portefeuille"][i] = round(nb_actions * pf_actions["V_bourse"][i], 2)
            pf_actions["Gains"][i] = round((((pf_actions["V_portefeuille"][i] - pf_actions["V_portefeuille"][i-1]) / pf_actions["V_portefeuille"][i])*100), 2)
            pf_actions["Cum_gains"][i] = round(pf_actions["Gains"][i] + pf_actions["Cum_gains"][i-1], 2)

        capital = capital + pf_actions["V_portefeuille"].iloc[-1]
        profit = pf_actions["V_portefeuille"].iloc[-1] - pf_actions["V_portefeuille"][0]

        return capital, pf_actions, profit
    
    def cross_moving_avr_strategy(self, strategy):
        """
        ...
        """
        # Récupération des variables
        capital = self.capital
        commission = self.commission

        # Définition des outils
        
        # Définition des signaux

        # 1. Test sur 2 ans pour identifier la meilleure action

        # 2. Execution de la stratégie sur le reste de la période


    
# %% Test 1 : Stratégie simple (market return)
# -- Strategie
Strategie1 = Agent(capital = 10000, commission = 0)
capital, pf_actions, profit = Strategie1.market_strategy()

# %%
# -- Informations
pf_actions.Cum_gains.plot()

# %% Test 2 : Cross Moving average - Court terme
# -- Strategie

# -- Informations

# %% Test 3 : Cross Moving average - Long terme
# -- Strategie

# -- Informations