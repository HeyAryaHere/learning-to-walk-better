class Enviorment:  # Defines a class named 'Environment' (likely a typo)
    def __init__(self):  # Defines the constructor (initialization method) for the Environment class
        self.actions = []  # Initializes an empty list named 'actions' to store actions taken
        self.states = []   # Initializes an empty list named 'states' to store states encountered
        self.logprobs = []  # Initializes an empty list named 'logprobs' to store log probabilities of actions (relevant for PPO)
        self.rewards = []   # Initializes an empty list named 'rewards' to store rewards received
        self.state_values = []  # Initializes an empty list named 'state_values' to store estimated state values (relevant for PPO)
        self.is_terminals = []  # Initializes an empty list named 'is_terminals' to store episode termination flags (True/False)

    def clear(self):  # Defines a method named 'clear'
        del self.actions[:]  # Deletes all elements from the 'actions' list
        del self.states[:]   # Deletes all elements from the 'states' list
        del self.logprobs[:]  # Deletes all elements from the 'logprobs' list
        del self.rewards[:]   # Deletes all elements from the 'rewards' list
        del self.state_values[:]  # Deletes all elements from the 'state_values' list
        del self.is_terminals[:]  # Deletes all elements from the 'is_terminals' list
