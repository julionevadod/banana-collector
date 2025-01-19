import copy
import collections
import numpy as np
import torch
from unityagents import UnityEnvironment
from .ValueFunction import ValueFunction
from .ExperienceReplayBuffer import ExperienceReplayBuffer

BUFFER_SIZE = 100000
WINDOW_SIZE=100
MAX_STEPS=10000
UPDATE_EVERY=4
STATE_SIZE=37
ACTION_SIZE=4
BRAIN_NAME="BananaBrain"
ENV_FILE="Banana.app"

device = "cuda:0" if torch.cuda.is_available() else "cpu"

class Agent:
    """_summary_
    """
    def __init__(self, gamma: float = 0.99, lr: float = 5e-4, eps_decay: float = 0.997, eps_end: float = 0.1):
        self.gamma = gamma
        self.eps = 1.0
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.loss = torch.nn.functional.mse_loss
        self.env = UnityEnvironment(file_name=ENV_FILE)
        self.replay_buffer = ExperienceReplayBuffer(
            BUFFER_SIZE
        )
        self.target_network = ValueFunction(
            STATE_SIZE,
            ACTION_SIZE
        ).to(
            device
        ).eval()
        self.local_network = ValueFunction(
            STATE_SIZE,
            ACTION_SIZE
        ).to(
            device
        ).eval()
        self.optimizer = torch.optim.Adam(self.local_network.parameters(), lr = lr)

    def _select_action(self, state):
        """Given input state, select an action based on current policy
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(
                state,
                device=device,
                dtype=torch.float64
            )
        if np.random.rand() > self.eps:
            with torch.no_grad():
                local_network_output = self.local_network(
                    state
                )
            action = local_network_output.detach().max().item() #TODO: Detach needed?
        else:
            action = np.random.randint(ACTION_SIZE)
        return action

    def _act(self): #step would be a better name
        """From a state A, select an action to switch to state B and get a reward
        """
        pass

    def _update(self, experiences):
        """Update policy from a batch of experiences
        """
        self.local_network.train()

        # Split experiences #TODO: manage better tensors
        states, actions, next_states, rewards, dones = zip(*experiences)
        states = torch.tensor(np.array(states),dtype=torch.float64).to(device)
        actions = torch.tensor(np.array(actions),dtype=torch.int64).to(device).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states),dtype=torch.float64).to(device)
        rewards = torch.tensor(np.array(rewards),dtype=torch.float64).to(device).unsqueeze(1)
        dones = torch.tensor(np.array(dones),dtype=torch.int64).to(device).unsqueeze(1)

        # Local network pass
        local_network_output = self.local_network(states)
        print(actions)
        local_expected_reward = local_network_output.gather(
            1, actions
        ) # TODO: why gather and not max?

        # Target network pass
        target_network_output = self.target_network(next_states).detach().max(
            dim = 1
        ).values.unsqueeze(1)*(1-dones) #TODO: why 1-dones

        target_expected_reward = rewards + self.gamma*target_network_output

        # Compute loss
        loss = self.loss(local_expected_reward,target_expected_reward)

        # Update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # self.soft_update(self.local_network,self.target_network,1e-3)
        self.target_network = copy.deepcopy(self.local_network)
        self.eps = max(self.eps*self.eps_decay,self.eps_end)
        self.local_network.eval()

    def learn(self, n_episodes: int, batch_size: int = 4):
        """Make agent learn how to interact with its given environment
        """
        scores = []
        scores_window = collections.deque(maxlen=WINDOW_SIZE)
        for i in range(n_episodes):
            env_info = self.env.reset(train_mode=True)[BRAIN_NAME]
            state = env_info.vector_observations[0],
            score = 0
            for j in range(MAX_STEPS):
                action = self._select_action(state)
                env_info = self.env.step(action)[BRAIN_NAME]
                next_state = env_info.vector_observations[0]
                reward = env_info.rewards[0]
                done = env_info.local_done[0]
                score += reward
                state = next_state
                self.replay_buffer.insert(
                    [state, action, next_state, reward, done]
                )
                if done:
                    break
                if (j%UPDATE_EVERY==0) & (len(self.replay_buffer)>=batch_size):
                    experiences = self.replay_buffer.sample(
                        batch_size
                    )
                    self._update(experiences)
            scores.append(score)
            scores_window.append(score)
            print("\rEPISODE {}/{}: Average Reward Last 100: {:.2f} \t Last Episode: {:.2f}".format(i,n_episodes,float(np.mean(scores_window)),score), end="")
            if i%100 == 0:
                print("\rEPISODE {}/{}: Average Reward Last 100: {:.2f} \t Last Episode: {:.2f}".format(i,n_episodes,float(np.mean(scores_window)),score))
        return scores


    def play(self):
        """Play an episode in the environment with the actual policy
        """
        pass


    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
