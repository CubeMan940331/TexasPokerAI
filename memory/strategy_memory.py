import random

class StrategyMemory:
    def __init__(self):
        self.memory = []

    def add(self, state_vector, strategy_vector):
        """Add a new strategy sample to memory."""
        self.memory.append((state_vector, strategy_vector))

    def sample(self, batch_size):
        """Randomly sample a batch of strategy samples."""
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def clear(self):
        """Clear all stored samples."""
        self.memory = []

    def __len__(self):
        return len(self.memory)
