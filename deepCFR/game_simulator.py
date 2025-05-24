def generate_random_info_state():
    # Stub: Replace with encoding of cards, position, pot, etc.
    return [0.1, 0.3, 0.5, 0.9]  # Example 4D state

def legal_actions():
    return ['fold', 'call', 'raise']  # Abstract actions

def simulate_episode():
    # Generates dummy info_states, regrets, and strategies
    return [
        (generate_random_info_state(), [0.1, -0.2, 0.1], [0.4, 0.4, 0.2])
        for _ in range(10)
    ]
