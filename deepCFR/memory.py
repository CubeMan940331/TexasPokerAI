import random

class Memory:
    def __init__(self):
        self.buffer = []

    def add(self, sample):
        self.buffer.append(sample)

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)
