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
        "move to state 2" : 0, 
        "move to state 4" : 0
    },
    "state 2": {
        "move to state 1" : 0, 
        "move to state 3" : 0, 
        "move to state 5" : 0
    },
    "state 3": {
        "move to state 2" : 0, 
        "move to state 6" : 0
    },
    "state 4": {
        "move to state 1" : 0,
        "move to state 5" : 0
    },
    "state 5": {
        "move to state 2" : 0,
        "move to state 4" : 0,
        "move to state 6" : 0
    },
    "state 6": {
        "move to state 3" : 0,
        "move to state"   : 0
    }
}


