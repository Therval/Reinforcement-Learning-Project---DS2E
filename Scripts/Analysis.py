#####################################################
## Agent simple: stratégies financières classiques ##
#####################################################
"""
Auteur : Valentin Joly
Date de création : 05/01/2021
Dernière modification : 23/01/21
"""

"""
Dans ce premier du projet de Reinforcement Learning, nous allons voir comment créer deux stratégies 
financières simples à des fins de comparaison avec un agent renforcé. 
Deux stratégies vont être abordées : 
    1 - Stratégie de marché : 
        Dans cette stratégie, l'agent va acheter un panier d'actions, à parts égales, pour se constituer un
        portefeuille dont les performances copieront exactement les mouvements du marché. 

    2 - Cross Moving Average : 
        Ici, l'agent possèdera deux indicateurs financiers : une moyenne mobile courte et une longue.
        Il va se donc s'en servir pour identifier les signaux d'achat et de vente. Le principe est très simple
        puisque lorsque la moyenne mobile la plus courte devient supérieure à la moyenne mobile la plus longue,
        alors l'agent achète. À l'inverse, c'est un signal de vente. 
"""

# %% 1. Chargement des librairies nécessaires et données
import pandas as pd
import numpy as np 
from tqdm import tqdm_notebook
import time
import matplotlib.pyplot as plt

# %% 2. Définition des strategies financières de bases
class Agent:
    """Framework de la classe Agent

    ----------
    Paramètres
    ----------
        Capital : Capital de départ de l'agent
        Commission : Commission appliquée à chaque passage d'ordre
    """
    def __init__(self, Capital, Commission):
        self.Capital = Capital # correspond à l'argent que l'agent peut investir
        self.Commission = Commission # Commission pour le passage d'un ordre
        self.stocks = pd.read_csv("/Users/valentinjoly/Documents/GitHub/Reinforcement-Learning-Project---DS2E/Data/Dataset_full.csv") #Représente le dataset 

        self.nb_actions = 0

    def market_strategy(self):
        """Définition de la stratégie de marché 

        ----------
        Paramètres
        ----------
            None
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
        for i in tqdm_notebook(range(1, len(self.stocks))):
            # Mise à jour du portefeuille d'actions
            pf_actions["V_bourse"][i] = round(self.stocks.loc[i, 'BNP.PA':'SAN.PA'].sum(), 2)
            pf_actions["Portefeuille"][i] = round(self.nb_actions * pf_actions["V_bourse"][i], 2)
        
        pf_actions["Gain"] = pf_actions["Portefeuille"].pct_change()*100
        pf_actions["Cum_gain"] = pf_actions.Gain.cumsum()

        self.Capital = self.Capital + pf_actions["Portefeuille"].iloc[-1]
        profit = pf_actions["Portefeuille"].iloc[-1] - pf_actions["Portefeuille"][0]
        
        print("Profit réalisé par cet agent: " + str(round(profit, 2)) + "€")
        print("Gain cumulé: " + str(round(pf_actions["Cum_gain"].iloc[-1], 2)) + "%")

        return self.Capital, pf_actions, profit
    



    def cross_moving_avr_strategy(self, ma_court, ma_long):
        """"Définition de la stratégie Cross Moving Average

        ----------
        Paramètres
        ----------
            ma_court : Nombre de jours utilisés pour calculer la moyenne mobile la plus courte
            ma_long : Nombre de jours utilisés pour calculer la moyenne mobile la plus longue
        """
        # Calcul de volatilité pour choix de l'action à trade (test sur 2 ans)
        variation_jour = self.stocks.iloc[0:538].pct_change()
        # Quelle action maximise la volatilité ?
        max_vola = variation_jour.rolling(50).std() * np.sqrt(50)
        action = max_vola.sum().idxmax(axis = 1)
        print("L'action qui maximise la volatilité sur la période est " + str(action))

        # 2. Exécution de la stratégie sur le reste de la période
        # Nouveau dataset avec l'action identifiée seulement
        stock = pd.DataFrame(self.stocks[action].iloc[539:5381])
        stock = stock.reset_index() # réinitialiser l'index du dataset
        stock.drop(["index"], axis=1, inplace = True)

        # Calcul des moyennes mobiles
        stock['MA_court'] = stock.iloc[:, 0].rolling(window=ma_court, min_periods=1, center=False).mean()
        stock['MA_long'] = stock.iloc[:, 0].rolling(window=ma_long, min_periods=1, center=False).mean()

        # Définition des variables de signaux d'informations
        stock["Signal"] = np.where(stock['MA_court'] > stock['MA_long'], 1, 0)   
        stock["Ordre"] = stock["Signal"].diff()

        # Définition des variables d'évolution du portefeuille, capital et gains 
        Portefeuille = [0]
        Capital = [self.Capital]
        Gains = [0]

        # Execution de la strategy
        for i in tqdm_notebook(range(1, len(stock))):
            # Signal d'achat 1, vente -1
            if stock["Ordre"][i] == 1:
                "Dans ce cas, on achète"
                self.nb_actions = int((Capital[i-1] - Capital[i-1]*self.Commission) / stock[action][i]) #Nombre d'action que l'on peut acheter
                Portefeuille.append(round((self.nb_actions*stock[action][i]), 2)) 
                Capital.append(round(Capital[i-1] - (Portefeuille[i] + (Portefeuille[i]*self.Commission)), 2))

            elif stock["Ordre"][i] == -1:
                Capital.append(((Portefeuille[i-1] - Portefeuille[i-1]*self.Commission) + Capital[i-1]))
                Portefeuille.append(0)
                self.nb_actions = 0

            else:
                Portefeuille.append(round((self.nb_actions*stock[action][i]), 2))
                Capital.append(Capital[i-1])
        
        # Ajout de variables au dataset pour analyse
        stock["Portefeuille"] = Portefeuille
        stock["Capital"] = Capital
        stock["Total"] = stock["Portefeuille"] + stock["Capital"]
        stock["Gain"] = stock["Total"].pct_change()*100
        stock["Cum_gain"] = stock.Gain.cumsum()

        profit = stock["Total"].iloc[-1] - stock["Total"].iloc[0] # Calcul du profit

        print("Profit réalisé par cet agent: " + str(round(profit, 2)) + "€")
        print("Gain cumulé: " + str(round(stock["Cum_gain"].iloc[-1], 2)) + "%")

        return stock, profit


# %% Test 1 : Stratégie simple (market return)
# -- Strategie
Strategie1 = Agent(Capital = 25000, Commission = 0)
capital, pf_actions, profit = Strategie1.market_strategy()

# -- Informations
# Gain cumulatif
pf_actions.Cum_gain.plot(figsize = (16, 10))
plt.title("Gain cumulatif : Stratégie 1")

# %% Test 2 : Cross Moving average
# -- Strategie
Strategie2 = Agent(Capital = 25000, Commission = 0.01)
stock, profit = Strategie2.cross_moving_avr_strategy(ma_court = 50, ma_long=200)

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