######################################
## Agent renforcé : Deep Q-learning ##
######################################
"""
Autheur : Valentin Joly
Date de création : 05/01/2021
Dernière modification : 23/01/21
"""

# %% 1. Librairies
# Envrionnement
import pandas as pd
import random
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from tqdm import tqdm_notebook

# Réseau de neurones lié à l'agent 
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import Adam

# Rendu graphique en temps réel de l'agent
import matplotlib as mpl
import matplotlib.pyplot as plt
%matplotlib

# %% 2. Class definition
"""
Dans cette classe, nous allons définir à la fois l'environnement mais aussi l'agent.
On se sert de la classe 'gym.Env' comme structure par défaut pour faciliter le développement 
du code et améliorer sa vitesse d'exécution. 

Des commentaires en-dessous chacune des fonctions de la classe sont inscrits pour faciliter 
la compréhension de celles-ci.
"""

class TradingEnvironnement(gym.Env):
    """Framework de la classe TradingEnvironnement.

    ----------
    Paramètres
    ----------
        windows_size : Taille de la fenêtre d'exécution (paramètre hérité de la classe mère gym.Env)
        Capital : Integer, Capital de départ de l'agent
        Commission : Integer, Commission prise pour le passage d'ordres
    """
    def __init__(self, windows_size, Capital, Commission):
        # Paramètres de l'agent 
        self.actions = {
        'Buy': np.array([1, 0, 0]),
        'Hold': np.array([0, 1, 0]),
        'Sell': np.array([0, 0, 1])
        }
        self.Capital_départ = Capital
        self.Capital = Capital
        self.Commission = Commission
        self.historique = []
        # Paramètres de l'envrionnement
        self.stock = self._load_csv()

        self.windows_size = windows_size
        self.n_features = self.stock.shape[1]
        self.shape = (self.windows_size, self.n_features+3)

        self.seed()

        # Definition de l'espace d'actions
        self.action_space = spaces.Discrete(len(self.actions)) # Permet de discétiser nos actions
        self.observation_space = spaces.Box(low=np.inf, high=np.inf, shape=self.shape, dtype=np.float32)
    

    def _load_csv(self, ma_court = 50, ma_long = 200):
        """Chargement du dataset et ajout des indicateurs financiers (moyennes mobiles)
            Attention : Si execution de l'agent dans un environnement Windows, le lien renvoyant
            au dataset doit être de la forme : r"path"

        ----------
        Paramètres
        ----------
            ma_court : Nombre de jours à prendre en compte pour le calcul de la moyenne mobile la plus courte
            ma_long : Nombre de jours à prendre en compte pour le calcul de la moyenne mobile la plus longue
        """
        df = pd.read_csv("/Users/valentinjoly/Documents/GitHub/Reinforcement-Learning-Project---DS2E/Data/Dataset_full.csv")

        # Selection de l'action MC.PA (pour comparaison avec les autres modèles non renforcés)
        stock = pd.DataFrame(df["MC.PA"].copy())
        # Ajout des moyennes mobiles
        stock['MA_court'] = stock.iloc[:, 0].rolling(window=ma_court, min_periods=1, center=False).mean()
        stock['MA_long'] = stock.iloc[:, 0].rolling(window=ma_long, min_periods=1, center=False).mean()

        return stock


    def seed(self, seed = 123):
        """Fonction permettant la reproducibilité du processus (utilisation dans la fonction reset(.))

        ----------
        Paramètres
        ----------
            seed : Integer 
        """
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def render(self, episode, update):
        """Permet la viusalisation du processus de trading de l'agent

        Attention, le rendu est particulièrement gourmand en mémoire RAM et peut ralentir fortement 
        la rapidité d'exécution du script. 
        Il est préférable de ne pas l'utiliser lors de l'entraînement de l'agent, sauf à des fins de 
        debugging et avec une mise à jour du graphiques tous les 50 jours.

        ----------
        Paramètres
        ----------
            episode : Permet d'afficher sur le graphique dynamique l'épisode en cours (entraînement de l'agent)
            update : Mise à jour du rendu graphique (nombre de jours)
        """
        # Initialisation du graphique lors de la première observation d'entraînement
        if self._first_render:
            self._f, self._ax = plt.subplots(2, 1, sharex=True)
            self._f.subplots_adjust(hspace = 0.3)

            self.j_graph, self.prix_graph, self.profit_graph = [], [], []
            self.ma1_graph, self.ma2_graph, self.actions_graph = [], [], []

            self._f.set_size_inches(12, 6)
            self._first_render = False
        
        # Fonction permettant d'afficher correctement la légende avec un graphique dynamique
        def legend_labels(ax):
            handles, labels = ax.get_legend_handles_labels()
            unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
            ax.legend(*zip(*unique), loc = "upper left")
        
        # Mise à jour des valeurs
        self.j_graph.append(self.jour)
        self.prix_graph.append(self.stock["MC.PA"][self.jour])
        self.ma1_graph.append(self.stock["MA_court"][self.jour])
        self.ma2_graph.append(self.stock["MA_long"][self.jour])
        
        self.actions_graph.append(self.action)

        self.profit_graph.append(self.profit)

        if self.jour % int(update) == 0:
            # Affichage du graphique lié au prix, moyennes mobiles et actions de l'agent
            self._ax[1].plot(self.j_graph, self.prix_graph, color='royalblue', linewidth = 4, alpha = 0.5, 
                            label = "Prix") # Plot prix
            self._ax[1].plot(self.j_graph, self.ma1_graph, color='yellowgreen', linewidth = 2, linestyle = 'dashed', 
                            label = "MA court")
            self._ax[1].plot(self.j_graph, self.ma2_graph, color='orchid', linewidth = 2, linestyle = 'dashed', 
                            label = "MA long")
                # Plot actions
            for i in range(0, len(self.actions_graph)):
                if all(self.actions_graph[i] == self.actions['Sell']):
                    self._ax[1].scatter(self.j_graph[i], self.prix_graph[i], 
                                            color='r', marker='v', linewidth = 8)
                elif all(self.actions_graph[i] == self.actions['Buy']):
                    self._ax[1].scatter(self.j_graph[i], self.prix_graph[i], 
                                            color='g', marker='^', linewidth = 8)

            # Mise à jour des axes 
            self._ax[1].set_xlim(0, self.jour + 20)
            self._ax[1].set_ylim(0, max(self.stock["MC.PA"][0:(self.jour + 50)] + 30))

            # Affichage du graphique d'évolution du profit de l'agent
            self._ax[0].plot(self.j_graph, self.profit_graph, color='blue', label = "Profit") # Plot profit
            self._ax[0].set_ylim(min(self.profit_graph) - 3000, max(self.profit_graph) + 3000)

            # Mise en page finale avant visualisation 
            self._f.tight_layout()
            plt.suptitle('Capital: ' + "%.2f" % self.Capital + ' ~ ' +
                        'Portefeuille: ' + "%.2f" % self.Portefeuille + ' ~ ' +
                        'Episode: ' +  str(episode))
            
            legend_labels(self._ax[0])
            legend_labels(self._ax[1])

            plt.pause(0.000001)


    def reset(self, Capital):
        """Permet l'initialisation de l'environnement en t = 0

        ----------
        Paramètres
        ----------
            Capital : Integer, Capital de départ de l'agent
        """
        # Initialisation en jour 0
        self.jour = 0 
        # Action de départ = Hold
        self.action = self.actions['Hold']

        # Variables liées à l'agent
        self.historique = []
        self.Capital = Capital
        self.Portefeuille = 0.0
        self.profit = 0.0
        self.have_position = False # on démarre avec 0 position sur le marché
        self.reward = 0.0

        self._first_render = True # initialiser le rendu dynamique 

        self.done = False # État de l'entraînement, si True --> Arrêt du programme

        self.updateState() # Production de l'environnement en t=0

        return self.state


    def step(self, action):
        """Fonction principale de l'agent. Elle permet de définir, en fonction de l'action qu'il choisit,
            les différentes conséquences sur les variables qui lui sont liées (Capital, Portefeuille, profit...)

        ----------
        Paramètres
        ----------
            action : Array / list acceptée également
        """

        if self.done:
            return self.state, self.reward, self.done
        
        self.action = action

        # Descriptif des actions
        if all(action == self.actions["Buy"]):
            if not self.have_position:
                if self.Capital > self.stock["MC.PA"][self.jour] + self.Capital*self.Commission:
                    self.have_position = True
                    self.Capital_achat = self.Capital # Enregistrer pour utilisation lors de la vente (mise à jour du profit)

                    prix_achat = round(self.stock["MC.PA"][self.jour], 3)
                    titres_achetés = int((self.Capital - self.Capital*self.Commission) / prix_achat)
                    commission_achat = self.Capital*self.Commission
                    self.Portefeuille += titres_achetés * prix_achat

                    self.Capital -= self.Portefeuille + commission_achat

                    self.reward -= commission_achat
                else:
                    self.done = True # Banqueroute --> Stop

            elif self.have_position:
                variation = ((self.stock["MC.PA"][self.jour] - self.stock["MC.PA"][self.jour-1]) / self.stock["MC.PA"][self.jour-1])
                self.Portefeuille += self.Portefeuille*variation # Mise à jour du portefeuille
                self.action = self.actions["Hold"]

        elif all(action == self.actions["Sell"]):
            if self.have_position:
                commission_vente = self.Portefeuille*self.Commission

                self.Capital += (self.Portefeuille - commission_vente)
                self.Portefeuille = 0
                self.have_position = False

                self.reward += (self.Capital - self.Capital_achat)
                self.profit += (self.Capital - self.Capital_achat)

            elif not self.have_position:
                if self.jour > 0:
                    variation = ((self.stock["MC.PA"][self.jour] - self.stock["MC.PA"][self.jour-1]) / self.stock["MC.PA"][self.jour-1])
                    self.reward += ((self.Capital*variation) / 10) # Pénalité pour ne rien faire (pénalité morale)
                    self.action = self.actions["Hold"]
        

        elif all(action == self.actions["Hold"]):
            if self.have_position:
                variation = ((self.stock["MC.PA"][self.jour] - self.stock["MC.PA"][self.jour-1]) / self.stock["MC.PA"][self.jour-1])
                self.Portefeuille += self.Portefeuille*variation 

            elif not self.have_position:
                if self.jour > 0:
                    variation = ((self.stock["MC.PA"][self.jour] - self.stock["MC.PA"][self.jour-1]) / self.stock["MC.PA"][self.jour-1])
                    self.reward += ((self.Capital*variation) / 10)

        self.jour += 1
        self.historique.append((self.stock.iloc[self.jour], self.Capital, self.Portefeuille, self.profit, self.reward, self.action))
        
        # On met à jour l'environnement avec les valeurs suivantes (prix, moyennes mobiles en t+1)
        self.updateState()

        if (self.jour > (self.stock.shape[0] - self.windows_size-1)):
            self.done = True

        return self.state, self.reward, self.done 
        

    def updateState(self):
        """
        Permets la mise à jour de l'état de l'environnement avec les informations nécessaires
        Fonction utilisée lors de la boucle d'entraînement

        ----------
        Paramètres
        ----------
            None
        """
        # Regroupement des informations de l'environnement dans un vecteur
        self.state = np.concatenate((self.stock.iloc[self.jour], self.action, [self.reward])) 

        return self.state

# %%
class ReinforcedAgent:
    """Framework de l'agent de type DQN (Deep Q-network)

        ----------
        Paramètres
        ----------
            state_size : Taille de l'environnement (se référer à self.observation_space de la classe ReinforcedAgent)
            action_size : Taille du vecteur des actions (self.action_size)
            episodes : Nombre d'épisodes pour l'entraînement de l'agent
            episode_length : Nombre d'observations à utiliser pour l'entraînement de l'agent, par épisode
            memory_size : Taille de la mémoire de l'agent
            train_interval : Correspond à l'interval où l'agent corrige la valeur d'epsilon
            gamma : Discount facteur des rewards futurs
            learning_rate : Learning rate du réseau de neurones
            batch_size : Nombre d'observations pour entraîner le modèle de l'agent
            epsilon_min : Valeur minimale qu'epsilon peut prendre

    """
    def __init__(self,
                 state_size,
                 action_size,
                 episodes,
                 episode_length,
                 memory_size=2000,
                 train_interval=100,
                 gamma=0.95,
                 learning_rate=0.001,
                 batch_size=64,
                 epsilon_min=0.01
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory_size = memory_size
        self.memory = [None] * memory_size
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_min = epsilon_min
        self.epsilon_decrement = 0.995
        self.learning_rate = learning_rate
        self.train_interval = train_interval
        self.batch_size = batch_size
        self.model = self.model()
        self.i = 0

        self.episodes = episodes
        self.episodes_length = episode_length

    def model(self):
        """Construction du réseau de neurones

        ----------
        Paramètres
        ----------
            None
        """
        model= Sequential()
        activation = "tanh"
        model.add(Dense(64,
                        input_dim=self.state_size,
                        activation=activation))
        model.add(Dense(64, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation=activation))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    def act(self, state):
        """Politique que suit l'agent. Corresponds à la politique epsilon-greedy.

        ----------
        Paramètres
        ----------
            state : corresponds à l'environnement ainsi qu'aux variables qui lui sont liées (prix, moyennes 
                    mobiles, l'action de l'agent et les rewards)
        """
        action = np.zeros(self.action_size)
        if np.random.rand() <= self.epsilon:
            action[random.randrange(self.action_size)] = 1
        else:
            state = state.reshape(1, self.state_size)
            act_values = self.model.predict(state)
            action[np.argmax(act_values[0])] = 1
        return action

    def observe(self, state, action, reward, next_state, done, warming_up=False):
        """Management de la mémoire de l'agent, mise à jour de la valeur d'epsilon et prédiction 
            de la meilleure action à prendre, en fonction de l'état actuel, à l'aide du réseau de neurones. 

        ----------
        Paramètres
        ----------
            state : correspond à l'environnement
            action : Action que l'agent à choisi
            reward : Reward de l'agent (ne correspond pas à la variable self.profit)
            next_state : État prochain
            done : Booléan
            warming_up : Paramètre de contrôle pour mettre à jour espilon 
        """
        self.i = (self.i + 1) % self.memory_size
        self.memory[self.i] = (state, action, reward, next_state, done)

        if (not warming_up) and (self.i % self.train_interval) == 0:
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decrement
            state, action, reward, next_state, done = self._get_batches()
            reward += (self.gamma
                       * np.logical_not(done)
                       * np.amax(self.model.predict(next_state),
                                 axis=1))
            q_target = self.model.predict(state)
            q_target[action[0], action[1]] = reward

            return self.model.fit(state, q_target,
                                  batch_size=self.batch_size,
                                  epochs=1,
                                  verbose=False,
                                  )

    def _get_batches(self):
        """sélectionne un "batch" de la mémoire de l'agent pour le réentraîner sur des données prises aléatoirement.
            Permet d'éviter que l'agent overfit les données et apprenne parfaitement la série temporelle. 

        ----------
        Paramètres
        ----------
            None
        """
        batch = np.array(random.sample(self.memory, self.batch_size))
        state_batch = np.concatenate(batch[:, 0])\
            .reshape(self.batch_size, self.state_size)
        action_batch = np.concatenate(batch[:, 1])\
            .reshape(self.batch_size, self.action_size)
        reward_batch = batch[:, 2]
        next_state_batch = np.concatenate(batch[:, 3])\
            .reshape(self.batch_size, self.state_size)
        done_batch = batch[:, 4]

        # Processing de l'action
        action_batch = np.where(action_batch == 1)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch


# %%
if __name__ == "__main__":
    # Instancier l'environnement
    Capital = 25000
    env = TradingEnvironnement(windows_size = 30, Capital = Capital, Commission = 0.01)

    # Instancier l'agent
    agent = ReinforcedAgent(state_size=len(env.reset(Capital = Capital)),
                            action_size=len(env.actions),
                            memory_size=50000,
                            episodes=20,
                            episode_length=len(env.stock),
                            train_interval=50,
                            gamma=0.1,
                            learning_rate=0.003,
                            batch_size=32,
                            epsilon_min=0.05)

    # "Échauffement" de l'agent
    for _ in range(agent.memory_size):
        action = agent.act(env.reset(Capital = Capital))
        next_state, reward, done = env.step(action)
        agent.observe(env.reset(Capital = Capital), action, reward, next_state, done, warming_up=True)

    # Entraînement de l'agent
    for ep in tqdm_notebook(range(agent.episodes)):
        state = env.reset(Capital = Capital)
        rew = 0
        for _ in range(agent.episodes_length):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # env.render(episode = ep, update = 100) # Optionnel, uniquement si debugging
            loss = agent.observe(state, action, reward, next_state, done)
            state = next_state
            rew += reward

    # Test de l'agent 
    done = False
    state = env.reset(Capital = Capital)
    while not done:
        action = agent.act(state)
        state, reward, done = env.step(action)
        env.render(episode = 0, update = 20)