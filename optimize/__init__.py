from collections import defaultdict

import pandas as pd

from optimize.greedy import run_greedy_optimization
from optimize.utilities import suggest_squad_roles
from optimize.utilities import get_future_gameweeks
from optimize.milp import solve
from optimize.parameters import get_parameter


def optimize_squad(
    current_season: str, 
    current_squad: set, 
    current_budget: int, 
    free_transfers: int,
    next_gameweek: int, 
    wildcard_gameweeks: list,
    predictions: pd.Series,
    elements: pd.DataFrame, 
    now_costs: pd.Series, 
    selling_prices: pd.Series, 
):

    return milp_optimization(
        current_season, current_squad, current_budget, free_transfers,
        next_gameweek, wildcard_gameweeks, predictions, elements, now_costs, selling_prices
    )


def milp_optimization(
    current_season: str, 
    current_squad: set, 
    current_budget: int, 
    free_transfers: int,
    next_gameweek: int, 
    wildcard_gameweeks: list,
    predictions: pd.Series,
    elements: pd.DataFrame, 
    now_costs: pd.Series, 
    selling_prices: pd.Series, 
):
    
    gameweeks = get_future_gameweeks(next_gameweek, wildcard_gameweeks=wildcard_gameweeks)

    if (current_season == '2022-23') and (7 in gameweeks):
        gameweeks.remove(7)

    # Prepare input for the solver
    players = list(elements.index)
    teams = elements['team'].to_dict()
    element_types = elements['element_type'].to_dict()
    points = defaultdict(lambda: 0, predictions.to_dict())
    now_costs = now_costs.to_dict()
    selling_prices = selling_prices.to_dict()

    for p in players:
        if p not in selling_prices:
            selling_prices[p] = now_costs[p]

    solution = solve(
        gameweeks,
        players,
        teams,
        element_types,
        points,
        current_squad,
        current_budget,
        free_transfers,
        now_costs,
        selling_prices,
        wildcard_gameweeks,
        get_parameter('squad_evaluation_round_factor'),
        get_parameter('starting_xi_multiplier'),
        get_parameter('captain_multiplier'),
        get_parameter('vice_captain_multiplier'),
        get_parameter('reserve_gkp_multiplier'),
        get_parameter('reserve_out_multiplier')[0],
        get_parameter('reserve_out_multiplier')[1],
        get_parameter('reserve_out_multiplier')[2],
    )

    return solution[next_gameweek]


def greedy_optimization(
    current_season: str, 
    current_squad: set, 
    current_budget: int, 
    free_transfers: int,
    next_gameweek: int, 
    wildcard_gameweeks: list,
    gameweek_predictions: pd.Series,
    gameweek_elements: pd.DataFrame, 
    now_costs: pd.Series, 
    selling_prices: pd.Series, 
    transfers_made: int = 0
):

    future_gameweeks = get_future_gameweeks(next_gameweek, wildcard_gameweeks=wildcard_gameweeks)

    if (current_season == '2022-23') and (7 in future_gameweeks):
        future_gameweeks.remove(7)

    if next_gameweek in {1, *wildcard_gameweeks}:
        free_transfers = float('inf')

    best_squad = run_greedy_optimization(
        current_squad, current_budget, future_gameweeks, 
        now_costs, gameweek_elements, selling_prices, gameweek_predictions,
        free_transfers, transfers_made
    )

    best_roles = suggest_squad_roles(
        best_squad, 
        gameweek_elements['element_type'], 
        gameweek_predictions.loc[:, next_gameweek]
    )

    solution = {
        'squad': best_squad,
        **best_roles
    }

    return solution