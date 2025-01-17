UPDATE_EVERY = 4

class Agent:
    """_summary_
    """
    def __init__(self):
        pass

    def _select_action(self):
        """Given input state, select an action based on current policy
        """
        pass

    def _act(self): #step would be a better name
        """From a state A, select an action to switch to state B and get a reward
        """
        pass

    def _update(self):
        """Update policy from a batch of experiences
        """
        pass

    def learn(self, n_episodes: int, batch_size: int = 4):
        """Make agent learn how to interact with its given environment
        """
        pass

    def play(self):
        """Play an episode in the environment with the actual policy
        """
        pass