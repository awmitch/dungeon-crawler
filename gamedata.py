from settings import BATCH_SIZE, MIN_DATA
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import random
random.seed(43)
import math
from collections import deque, namedtuple

import logging

# Set up logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Neural Network for the Agent
class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, history_length=3):
        super(DQNNetwork, self).__init__()
        # Adjust input dimension to account for history
        self.input_dim = input_dim + history_length + 1
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))



class CentralizedDQN:
    def __init__(self):
        self.action_dim = 4
        self.state_dim = 8  # x, y, vx, vy for enemy and x, y, vx, vy for player as it was originally
        self.policy_net = DQNNetwork(self.state_dim, self.action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters())

        self.gamma = 0.99
        self.epsilon = 0.9
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        self.epsilon_max = 0.9  # Fix epsilon decay
        self.criterion = nn.MSELoss()
        self.global_reward = {}

        self.memory = deque(maxlen=10000)  # more efficient than a list
        self.target_net = DQNNetwork(self.state_dim, self.action_dim)  # For Soft Updates
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_update = 10  # Update target network every 10 optimizations

        self.action_history_length = 3
        self.action_history = {}  # Make it a dictionary with enemy_id as key
        self.last_action = {}     # Make it a dictionary with enemy_id as key
        self.action_repeat_count = {}  # Make it a dictionary with enemy_id as key

    def init_enemy(self, enemy_id):
        if enemy_id not in self.action_history:
            self.action_history[enemy_id] = deque([0]*self.action_history_length, maxlen=self.action_history_length)
            self.last_action[enemy_id] = None
            self.action_repeat_count[enemy_id] = 3
    
    # New method to set the global reward
    def set_reward(self, enemy_id, value):
        self.global_reward["%d"%enemy_id] = value

    # New method to get the current global reward
    def get_reward(self, enemy_id):
        return self.global_reward["%d"%enemy_id]
    
    # New method to reset the current global reward
    def reset_reward(self, enemy_id):
        self.global_reward["%d"%enemy_id] = 0

    # New method to add value to the global reward
    def add_to_reward(self, enemy_id, value):
        self.global_reward["%d"%enemy_id] += value

    def select_action(self, state, enemy_id):
        self.init_enemy(enemy_id)
        
        sample = random.random()

        # Incorporating action history into state
        extended_state = torch.cat((state, torch.tensor(self.action_history[enemy_id], dtype=torch.float32).unsqueeze(0)), dim=1)
        # Update conditions to use enemy_id:
        if self.action_repeat_count[enemy_id] > 0 and self.last_action[enemy_id] is not None:
            self.action_repeat_count[enemy_id] -= 1
            return self.last_action[enemy_id]

        if sample < self.epsilon:
            action = torch.tensor([[random.randrange(self.action_dim)]], dtype=torch.long)
        else:
            with torch.no_grad():
                action = self.policy_net(extended_state).max(1)[1].view(1, 1)  # Returns the action with highest Q-value

        # Update action history and last action
        self.action_history[enemy_id].append(action.item())
        self.last_action[enemy_id] = action
        self.action_repeat_count[enemy_id] = 3

        return action

    def store_transition(self, enemy_id, current_state, action, reward, next_state):
        # Adjust reward for switching directions
        action_hist = self.action_history[enemy_id]
        if action_hist[0] == 0 and action_hist[1] == 1 or action_hist[0] == 2 and action_hist[1] == 3:
            reward -= 1  # Decrease reward for changing direction
        
        # Store transition
        self.memory.append((current_state, action, reward, next_state))
        if len(self.memory) > 10000:  # limit memory size to 10,000
            del self.memory[0]
            
    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        batch_state = torch.cat(batch.state)
        batch_action = torch.cat(batch.action)
        batch_reward = torch.tensor(batch.reward, dtype=torch.float32)
        batch_next_state = torch.cat(batch.next_state)

        # Extend states with action history
        extended_batch_state = self.extend_state_with_history(batch_state)
        extended_batch_next_state = self.extend_state_with_history(batch_next_state)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(extended_batch_state).gather(1, batch_action)

        # Compute V(s_{t+1}) for all next states with Double DQN:
        selected_actions = self.policy_net(extended_batch_next_state).max(1)[1]
        next_state_values = self.target_net(extended_batch_next_state).gather(1, selected_actions.unsqueeze(-1)).squeeze(-1).detach()

        # Compute expected Q values
        expected_state_action_values = batch_reward + self.gamma * next_state_values

        # Use MSE Loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))
       
        # Compute Huber loss (smooth mean squared error)
        #loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)  # Clip gradients to stabilize training
        self.optimizer.step()

        # Soft Update with smaller tau
        tau = 0.001
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(tau * policy_param.data + (1.0 - tau) * target_param.data)


    def decay_epsilon(self, episode):
        self.epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) * math.exp(-self.epsilon_decay * episode)

    def extend_state_with_history(self, states):
        """
        Extend a batch of states with action history.
        """
        # Modify this to use the correct action history for each state in the batch
        batch_action_history = [list(self.action_history[i.item()]) for i in states[:, -1]] # Assuming enemy_id is the last element in the state
        batch_action_history = torch.tensor(batch_action_history, dtype=torch.float32).to(states.device)
        extended_states = torch.cat((states, batch_action_history), dim=1)

        return extended_states

class DQNSingleton:
    _instance = None
    dqn = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.dqn = CentralizedDQN()
        return cls._instance

dqn_singleton = DQNSingleton().dqn

class GameData:
    def __init__(self):
        self.damage_dealt = 0
        self.damage_taken = 0
        self.rooms_cleared = 0
        self.time_alive = 0
        self.sessions_played = -1

    def reset(self):
        self.damage_dealt = 0
        self.damage_taken = 0
        self.rooms_cleared = 0
        self.time_alive = 0
        self.sessions_played += 1

    def update_data(self, damage_dealt=0, damage_taken=0, rooms_cleared=0, time_alive=None):
        self.damage_dealt += damage_dealt
        self.damage_taken += damage_taken
        self.rooms_cleared += rooms_cleared
        if time_alive is not None:
            self.time_alive = time_alive

    def get_features(self):
        efficiency = (self.damage_dealt - self.damage_taken) / max(self.rooms_cleared, 1)
        #time_efficiency = self.time_alive / max(self.sessions_played, 1)
        return [self.damage_dealt, self.damage_taken, self.rooms_cleared, self.time_alive, efficiency]

    def get_average(self):
        if self.sessions_played == 0:
            return self.get_features()
        return [feature/self.sessions_played for feature in self.get_features()]


class DifficultyScaler:
    def __init__(self):
        self.model_fire_rate = RandomForestRegressor(n_estimators=MIN_DATA)
        self.model_speed = RandomForestRegressor(n_estimators=MIN_DATA)
        self.model_health = RandomForestRegressor(n_estimators=MIN_DATA)
        self.data_collected = []
        self.y_fire_rates = []
        self.y_speeds = []
        self.y_healths = []
        self.trained = False
        self.scaler_fire_rate= StandardScaler()  # Separate scalers
        self.scaler_speed = StandardScaler()
        self.scaler_health = StandardScaler()
        self.game_data = GameData()
        self.adjusted_fire_rate = 1.0
        self.adjusted_speed = 1.0
        self.adjusted_health = 1.0
        # Initial multipliers for each stat
        self.global_multiplier_fire_rate = 1.0
        self.global_multiplier_speed = 1.0
        self.global_multiplier_health = 1.0

    def compute_performance_score(self):
        game_data = self.game_data.get_features()
        damage_efficiency = game_data[0] / (game_data[1] + 1)  # How efficiently the player deals damage
        damage_ratio = game_data[0] / (game_data[1] + 1)  # Comparison of damage dealt vs taken
        progression_rate = game_data[2]  # The number of rooms cleared as an indicator of progression
        score = [damage_efficiency, damage_ratio, progression_rate]
        logger.debug("Performance Score: %s", score)
        return score

    
    def label_data(self, performance_scores):
        damage_efficiency, damage_ratio, progression_rate = performance_scores

        MAX_ADJUSTMENT = 1.2

        # To emphasize health scaling in a smoother way
        health_scaling = min((progression_rate**0.4 / 20), MAX_ADJUSTMENT - 1) # Cap the adjustment

        # Adjusting labels for more subtle changes and balance
        y_fire_rate = 1.0 + min((damage_ratio - 1) * 0.03, MAX_ADJUSTMENT - 1)  # Adjust fire rate based on damage ratio
        y_speed = 1.0 + min((damage_ratio - 1) * 0.03, MAX_ADJUSTMENT - 1)  # Adjust speed slightly based on damage ratio
        y_health = 1.0 + health_scaling + min((damage_efficiency - 1) * 0.05, MAX_ADJUSTMENT - 1)  # Adjust health based on player's efficiency and progression

        return y_fire_rate, y_speed, y_health


    def collect_data(self):
        # Get average data from game
        average_data = self.game_data.get_average()
            
        # Compute performance score and its label
        ps = self.compute_performance_score()
        yfr, ys, yh = self.label_data(ps)

        # Store labels in their respective lists
        self.y_fire_rates.append(yfr)
        self.y_speeds.append(ys)
        self.y_healths.append(yh)
        
        self.data_collected.append(average_data)
            
        # Train the models after every MIN_DATA data points
        if len(self.data_collected) > MIN_DATA:  
            self.train_models()
            self.trained = True

            # Save the models and scalers
            self.save_models()

    def train_models(self):
        # Scale the data
        X = self.data_collected
        X_fire_rate = self.scaler_fire_rate.fit_transform(X)
        X_speed = self.scaler_speed.fit_transform(X)
        X_health = self.scaler_health.fit_transform(X)

        # Use stored labels for training
        y_fire_rate = self.y_fire_rates
        y_speed = self.y_speeds
        y_health = self.y_healths

        # Incremental training
        if self.trained:  
            self.model_fire_rate.n_estimators += 1
            self.model_speed.n_estimators += 1
            self.model_health.n_estimators += 1

        # Train the models
        self.model_fire_rate.fit(X_fire_rate, y_fire_rate)
        self.model_speed.fit(X_speed, y_speed)
        self.model_health.fit(X_health, y_health)

    def adjust_difficulty(self):
        # Check if models are trained
        if not self.trained:  
            # If not trained, use default adjustments
            self.reset()
            return

        # Get current game data
        current_data = self.game_data.get_average()

        # Transform data using scalers
        current_data_fire_rate = self.scaler_fire_rate.transform([current_data])
        current_data_speed = self.scaler_speed.transform([current_data])
        current_data_health = self.scaler_health.transform([current_data])

        # Predict adjustments using trained models
        adjusted_fire_rate = self.model_fire_rate.predict(current_data_fire_rate)[0]
        adjusted_speed = self.model_speed.predict(current_data_speed)[0]
        adjusted_health = self.model_health.predict(current_data_health)[0]

        # Introduce random noise to adjustments
        noise_factor_fire_rate = np.random.uniform(0.95, 1.05)
        noise_factor_speed = np.random.uniform(0.95, 1.05)
        noise_factor_health = np.random.uniform(0.95, 1.05)

        adjusted_fire_rate *= noise_factor_fire_rate
        adjusted_speed *= noise_factor_speed
        adjusted_health *= noise_factor_health
        logger.debug("Adjustments: %s", [adjusted_fire_rate, adjusted_speed, adjusted_health])
        # Ensure adjusted values are within certain limits
        self.adjusted_speed = min(max(adjusted_speed, 0.5), 2.0)
        self.adjusted_health = min(max(adjusted_health, 0.5), 2.0)
        self.adjusted_fire_rate = min(max(adjusted_fire_rate, 0.5), 2.0)

        # Update global multipliers
        self.global_multiplier_fire_rate *= self.adjusted_fire_rate
        self.global_multiplier_speed *= self.adjusted_speed
        self.global_multiplier_health *= self.adjusted_health
        logger.debug("Global adjustments: %s", [self.global_multiplier_fire_rate, self.global_multiplier_speed, self.global_multiplier_health])

        # Ensure multipliers don't go beyond reasonable bounds
        # self.global_multiplier_speed = min(max(self.global_multiplier_speed, 0.5), 3.0)
        # self.global_multiplier_health = min(max(self.global_multiplier_health, 0.5), 3.0)
        # self.global_multiplier_fire_rate = min(max(self.global_multiplier_fire_rate, 0.5), 3.0)

    def reset(self):
        self.game_data.reset()
        self.adjusted_fire_rate = 1
        self.adjusted_speed = 1
        self.adjusted_health = 1
        self.global_multiplier_speed = 1
        self.global_multiplier_health = 1
        self.global_multiplier_fire_rate = 1

    def save_models(self):
        joblib.dump(self.model_fire_rate, 'model_fire_rate.pkl')  # Save model for fire rate
        joblib.dump(self.model_speed, 'model_speed.pkl')
        joblib.dump(self.model_health, 'model_health.pkl')
        joblib.dump(self.scaler_fire_rate, 'scaler_fire_rate.pkl')
        joblib.dump(self.scaler_speed, 'scaler_speed.pkl')
        joblib.dump(self.scaler_health, 'scaler_health.pkl')


    def load_models(self):
        # Load models and scalers if they exist
        try:
            self.model_fire_rate = joblib.load('model_fire_rate.pkl')  # Load model for fire rate
            self.model_speed = joblib.load('model_speed.pkl')
            self.model_health = joblib.load('model_health.pkl')
            self.scaler_fire_rate = joblib.load('scaler_fire_rate.pkl')
            self.scaler_speed = joblib.load('scaler_speed.pkl')
            self.scaler_health = joblib.load('scaler_health.pkl')
            self.trained = True
        except FileNotFoundError:
            pass

    def average_difficulty_scaling(self):
        return (self.global_multiplier_speed * 
                self.global_multiplier_health * 
                self.global_multiplier_fire_rate) ** (1/3)


class DifficultyScalerSingleton:
    _instance = None
    diff_scaler = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.diff_scaler = DifficultyScaler()
        return cls._instance

diff_scaler_singleton = DifficultyScalerSingleton().diff_scaler