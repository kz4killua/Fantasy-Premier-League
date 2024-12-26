import numpy as np


parameters = {
    'squad_evaluation_round_factor': 1.0,
    'captain_multiplier': 2.0,
    'starting_xi_multiplier': 1.0,
    'vice_captain_multiplier': 1.0,
    'reserve_gkp_multiplier': 0.0,
    'reserve_out_multiplier': [0.0, 0.0, 0.0],
    'future_gameweeks_evaluated': 1,
    'budget_importance': 0,
    'transfer_aversion_factor': 1.0,
}


def get_parameter(name: str):
    return parameters[name]


def set_parameter(name: str, value):
    if name not in parameters:
        raise ValueError(f"Parameter '{name}' not found.")
    parameters[name] = value