"""
RL Prject : Analyse
"""

# %% 1. Chargement des librairies nécessaires et données
# Libraries
import pandas as pd
import numpy as np 
import time
import matplotlib.pyplot as plt

# %% 2. Definition des strategies financières de bases
class Agent:
    def __init__(self, Capital, Commission):
        self.Capital = Capital #Correspond à l'argent que l'agent peut investir
        self.Commission = Commission #Commission pour le passage d'un ordre
        self.stocks = pd.read_csv("/Users/valentinjoly/Documents/GitHub/Reinforcement-Learning-Project---DS2E/Data/Dataset_full.csv") #Représente le dataset 

        self.nb_actions = 0

    def market_strategy(self):
        """
        ...
        """
        # Définition des variables
        pf_actions = pd.DataFrame(np.zeros((self.stocks.shape[0], 2)),
                                  columns = ["V_bourse", "Portefeuille"])

        # Definition des variables en t = 0
        pf_actions["V_bourse"][0] = round(self.stocks.loc[0, 'BNP.PA':'SAN.PA'].sum(), 2)

        self.nb_actions = int(self.Capital / pf_actions["V_bourse"][0])

        pf_actions["Portefeuille"][0] = round(self.nb_actions * pf_actions["V_bourse"][0], 2)

        self.Capital = self.Capital - pf_actions["Portefeuille"][0] # ce qu'il reste après investissement

        # Execution
        for i in range(1, len(self.stocks)):
            # Mise à jour du portefeuille d'actions
            pf_actions["V_bourse"][i] = round(self.stocks.loc[i, 'BNP.PA':'SAN.PA'].sum(), 2)
            pf_actions["Portefeuille"][i] = round(self.nb_actions * pf_actions["V_bourse"][i], 2)
        
        pf_actions["Gain"] = pf_actions["Portefeuille"].pct_change()*100
        pf_actions["Cum_gain"] = pf_actions.Gain.cumsum()

        self.Capital = self.Capital + pf_actions["Portefeuille"].iloc[-1]
        profit = pf_actions["Portefeuille"].iloc[-1] - pf_actions["Portefeuille"][0]
        
        print("Profit réalisé par cet agent: " + str(profit) + "€")
        print("Gain cumulé: " + str(round(pf_actions["Cum_gain"].iloc[-1], 2)) + "%")

        return self.Capital, pf_actions, profit
    

    
    def cross_moving_avr_strategy(self, ma_court, ma_long):
        """
        ...
        """
        # Calcul de volatilité pour choix de l'action à trade (test sur 2 ans)
        variation_jour = self.stocks.iloc[0:538].pct_change()
        # Quel action maximise la volatilité ?
        max_vola = variation_jour.rolling(50).std() * np.sqrt(50)
        action = max_vola.sum().idxmax(axis = 1)
        print("L'action qui maximise la volatilité sur la période est " + str(action))

        # 2. Execution de la stratégie sur le reste de la période
        # Nouveau dataset avec l'action identitifiée seulement
        stock = pd.DataFrame(self.stocks[action].iloc[539:5381])
        stock = stock.reset_index() # Réinitialiser l'index du dataset
        stock.drop(["index"], axis=1, inplace = True)

        # Calcul des moyennes mobiles
        stock['MA_court'] = stock.iloc[:, 0].rolling(window=ma_court, min_periods=1, center=False).mean()
        stock['MA_long'] = stock.iloc[:, 0].rolling(window=ma_long, min_periods=1, center=False).mean()

        # Definition des variables de signaux d'informations
        stock["Signal"] = np.where(stock['MA_court'] > stock['MA_long'], 1, 0)   
        stock["Ordre"] = stock["Signal"].diff()

        # Definition des variables d'évolution du portfeuille, capital et gains 
        Portefeuille = [0]
        Capital = [10000]
        Gains = [0]

        # Execution de la strategie
        for i in range(1, len(stock)):
            # Signal d'achat 1, vente -1
            if stock["Ordre"][i] == 1:
                "Dans ce cas, on achète"
                self.nb_actions = int((Capital[i-1] - Capital[i-1]*self.Commission) / stock["MC.PA"][i]) #Nombre d'action que l'on peut acheter
                Portefeuille.append(round((self.nb_actions*stock["MC.PA"][i]), 2)) 
                Capital.append(round(Capital[i-1] - (Portefeuille[i] + (Portefeuille[i]*self.Commission)), 2))

            elif stock["Ordre"][i] == -1:
                Capital.append(((Portefeuille[i-1] - Portefeuille[i-1]*self.Commission) + Capital[i-1]))
                Portefeuille.append(0)
                self.nb_actions = 0

            else:
                Portefeuille.append(round((self.nb_actions*stock["MC.PA"][i]), 2))
                Capital.append(Capital[i-1])
        
        # Ajout de variables au dataset pour analyse
        stock["Portefeuille"] = Portefeuille
        stock["Capital"] = Capital
        stock["Total"] = stock["Portefeuille"] + stock["Capital"]
        stock["Gain"] = stock["Total"].pct_change()*100
        stock["Cum_gain"] = stock.Gain.cumsum()

        profit = stock["Total"].iloc[-1] - stock["Total"].iloc[0] # Calcul du profit

        print("Profit réalisé par cet agent: " + str(profit) + "€")
        print("Gain cumulé: " + str(round(stock["Cum_gain"].iloc[-1], 2)) + "%")

        return stock, profit



# %% Test 1 : Stratégie simple (market return)
# -- Strategie
Strategie1 = Agent(Capital = 10000, Commission = 0)
capital, pf_actions, profit = Strategie1.market_strategy()

# -- Informations
# Gain cumulatif
pf_actions.Cum_gain.plot(figsize = (16, 10))
plt.title("Gain cumulatif : Stratégie 1")

# %% Test 2 : Cross Moving average
# -- Strategie
Strategie2 = Agent(Capital = 10000, Commission = 0.001)
stock, profit = Strategie2.cross_moving_avr_strategy(ma_court = 50, ma_long=150)

# %%
# -- Information
# Gain cumulatif
stock.Cum_gain.plot(figsize = (16, 10))
plt.title("Gain cumulatif : Stratégie 2")

# %% 
# Passage d'ordre
# Initialize the plot figure
fig = plt.figure(figsize = (16, 10))

ax1 = fig.add_subplot(111,  ylabel='Prix en €')
stock["MC.PA"].plot(ax=ax1, color='black', lw=2.)
stock[['MA_court', 'MA_long']].plot(ax=ax1, lw=2.)
ax1.plot(stock.loc[stock.Ordre == 1].index, 
         stock["MA_court"][stock.Ordre == 1.0],
         '^', markersize=10, color='g')
ax1.plot(stock.loc[stock.Ordre == -1].index, 
         stock["MA_long"][stock.Ordre == -1],
         'v', markersize=10, color='r')

plt.title("Visualisation des ordres passés")
plt.show()

# %%
