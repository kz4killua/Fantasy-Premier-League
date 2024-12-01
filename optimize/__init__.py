import pandas as pd

from optimize.utilities import get_future_gameweeks, make_best_transfer
from optimize.greedy import run_greedy_optimization


def optimize_squad(
    current_season: str, current_squad: set, current_budget: int, 
    next_gameweek: int, wildcard_gameweeks: list,
    now_costs: pd.Series, selling_prices: pd.Series, 
    gameweek_elements: pd.DataFrame, gameweek_predictions: pd.Series,
    free_transfers: int, transfers_made: int
):
    """Make transfers to optimize the squad for the next gameweek."""

    future_gameweeks = get_future_gameweeks(next_gameweek, wildcard_gameweeks=wildcard_gameweeks)

    if (current_season == '2022-23') and (7 in future_gameweeks):
        future_gameweeks.remove(7)

    best_squad = run_greedy_optimization(
        current_squad, current_budget, future_gameweeks, 
        now_costs, gameweek_elements, selling_prices, gameweek_predictions,
        free_transfers, transfers_made
    )

    return best_squad