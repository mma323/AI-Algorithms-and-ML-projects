discount_factor : float = 0.9
learning_rate   : float = 0.1

reward_matrix : dict = {
    "state 1": 0,
    "state 2": 0,
    "state 3": -10,
    "state 4": 0,
    "state 5": 0,
    "state 6": 100
}

q_matrix : dict = {
    "state 1": {
        "state 2" : 0, 
        "state 4" : 0
    },
    "state 2": {
        "state 1" : 0, 
        "state 3" : 0, 
        "state 5" : 0
    },
    "state 3": {
        "state 2" : 0, 
        "state 6" : 0
    },
    "state 4": {
        "state 1" : 0,
        "state 5" : 0
    },
    "state 5": {
        "state 2" : 0,
        "state 4" : 0,
        "state 6" : 0
    },
    "state 6": {
        "state 3" : 0,
        "state 5" : 0
    }
}

def update_q_matrix(state, action):
    q_matrix[state][action] = (
        q_matrix[state][action] + 
        learning_rate * (
            reward_matrix[action] + 
            discount_factor * max(q_matrix[action].values()) - 
            q_matrix[state][action]
        )
    )

update_q_matrix("state 4", "state 5")
update_q_matrix("state 5", "state 6")
update_q_matrix("state 4", "state 5")
update_q_matrix("state 1", "state 4")


print(q_matrix["state 2"])
print(q_matrix["state 3"])
print(q_matrix["state 5"])