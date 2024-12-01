import pandas as pd

from optimize.utilities import (
    get_valid_transfers, make_best_transfer, calculate_budget, 
    update_selling_prices, count_transfers_made
)


def run_greedy_optimization(
    initial_squad: set, initial_budget: int, gameweeks: list, now_costs: pd.Series,
    elements: pd.DataFrame, selling_prices: pd.Series, gameweek_predictions: pd.Series,
    free_transfers: int, transfers_made: int
):
    """Optimizes a squad by iteratively making the best transfers till convergence."""

    current_squad = initial_squad.copy()
    current_budget = initial_budget

    # When wildcarding or on GW 1, replace each player with their cheapest alternative.
    if free_transfers == float('inf'):
        current_squad, current_budget, selling_prices = free_budget(
            current_squad, current_budget, elements, selling_prices, now_costs
        )

    # Iteratively make the single best transfer, until the squad converges.
    current_squad = make_transfers_until_convergence(
        current_squad, current_budget, elements, gameweeks,
        now_costs, selling_prices, gameweek_predictions, 
        free_transfers, transfers_made
    )

    return current_squad


def free_budget(initial_squad: set, initial_budget: int, elements: pd.DataFrame, selling_prices: pd.Series, now_costs: pd.Series):
    """Replace each player with their cheapest alternative to free up budget."""

    current_squad = initial_squad.copy()
    current_budget = initial_budget

    # Replace each player with their cheapest alternative to free up budget.
    for player in list(current_squad):

        valid_replacements = get_valid_transfers(
            current_squad, player, elements, selling_prices, current_budget
        )
        valid_replacements_costs = now_costs.loc[list(valid_replacements)]
        cheapest_replacement = valid_replacements_costs.idxmin()

        # Update the squad (and budget) accordingly.
        new_squad = current_squad - {player} | {cheapest_replacement}
        new_budget = calculate_budget(
            current_squad, new_squad, current_budget, selling_prices, now_costs
        )
        selling_prices = update_selling_prices(
            selling_prices, now_costs, current_squad, new_squad
        )

        current_squad, current_budget = new_squad, new_budget

    return current_squad, current_budget, selling_prices


def make_transfers_until_convergence(
    initial_squad: set, initial_budget: int, elements: pd.DataFrame, gameweeks: list[int], 
    now_costs: pd.Series, selling_prices: pd.Series, gameweek_predictions: pd.Series, 
    free_transfers: int, transfers_made: int
): 
    """Iteratively make the single best transfer, until the squad converges."""

    current_squad = initial_squad.copy()
    current_budget = initial_budget

    while True:
        new_squad = make_best_transfer(
            current_squad, current_budget, gameweeks, elements, 
            selling_prices, now_costs, gameweek_predictions,
            free_transfers, transfers_made
        )
        new_budget = calculate_budget(
            current_squad, new_squad, current_budget, selling_prices, now_costs
        )
        selling_prices = update_selling_prices(
            selling_prices, now_costs, current_squad, new_squad
        )
        transfers_made += count_transfers_made(current_squad, new_squad)
        
        if new_squad == current_squad:
            break
        else:
            current_squad = new_squad
            current_budget = new_budget

    return current_squad
