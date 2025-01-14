import pandas as pd

from optimize import optimize_squad
from optimize.utilities import calculate_points, calculate_budget, sum_player_points, update_purchase_prices, update_selling_prices, calculate_transfer_cost, count_transfers_made
from predictions import make_predictions, group_predictions_by_gameweek, weight_gameweek_predictions_by_availability
from simulation.utilities import make_automatic_substitutions, get_selling_prices, get_player_name
from simulation.loaders import load_simulation_purchase_prices, load_simulation_bootstrap_elements, load_simulation_features, load_simulation_true_results
from optimize.rules import MAX_FREE_TRANSFERS


def format_currency(amount: int):
    """Format game currency as a string."""
    return f"${round(amount / 10, 1)}"


def print_gameweek_report(
    gameweek: int, 
    initial_squad: set, 
    final_squad: set, 
    selected_roles: set, 
    automatic_roles: set, 
    final_budget: int, 
    initial_selling_prices: pd.Series, 
    final_purchase_prices: pd.Series,
    player_points: pd.Series, 
    positions: pd.Series, 
    elements: pd.DataFrame,
    gameweek_points: int, 
    free_transfers: int,
    transfer_cost: int,
    overall_points: int,
):
    """Print a detailed summary of activity in a simulated gameweek."""

    position_names = {1: 'GKP', 2: 'DEF', 3: 'MID', 4: 'FWD'}

    print()
    print("-----------------------")
    print(f"Gameweek {gameweek}: {gameweek_points} points")
    print("-----------------------")

    # Print the starting XI, including substitutions, names, and captaincy
    print("\nStarting XI")
    for player in sorted(automatic_roles['starting_xi'], key=lambda player: positions.loc[player]):
    
        if player not in selected_roles['starting_xi']:
            print("->", end=" ")
        else:
            print("  ", end=" ")

        if player == automatic_roles['captain']:
            print("(C)", end=" ")
        elif player == automatic_roles['vice_captain']:
            print("(V)", end=" ")
        else:
            print("   ", end=" ")

        print(
            position_names[positions.loc[player]],
            get_player_name(player, elements),
            f"[{sum_player_points([player], player_points)} pts]"
        )

    # Print the reserve players, indicating previous captains and vice-captains
    print("\nReserves")
    for player in [automatic_roles['reserve_gkp'], *automatic_roles['reserve_out']]:

        if player in selected_roles['starting_xi']:
            print("<-", end=" ")
        else:
            print("  ", end=" ")

        if player == selected_roles['captain']:
            print("(*C)", end=" ")
        elif player == selected_roles['vice_captain']:
            print("(*V)", end=" ")
        else:
            print("    ", end=" ")

        print(
            position_names[positions.loc[player]],
            get_player_name(player, elements),
            f"[{sum_player_points([player], player_points)} pts]"
        )

    # Print transfer activity and final budget
    print(f"\nTransfers ({transfer_cost} points) [Free transfers: {free_transfers}]")
    for player in (final_squad - initial_squad):
        print(f"-> {get_player_name(player, elements)} ({format_currency(final_purchase_prices.loc[player])})")
    for player in (initial_squad - final_squad):
        print(f"<- {get_player_name(player, elements)} ({format_currency(initial_selling_prices.loc[player])})")
    print(f"\nBank: {format_currency(final_budget)}")

    # Print overall points
    print(f"\nOverall points: {overall_points}")


def get_initial_team_and_budget(season: str):
    """Return an initial squad and budget for a given season."""

    if season == '2021-22':
        initial_squad = {389, 237, 220, 275, 233, 359, 485, 196, 43, 177, 40, 376, 14, 62, 348}
        initial_budget = 0
    elif season == '2022-23':
        initial_squad = {81, 100, 285, 306, 139, 448, 14, 283, 446, 374, 486, 381, 80, 28, 237}
        initial_budget = 45
    elif season == '2023-24':
        initial_squad = {352, 616, 206, 398, 209, 6, 501, 294, 303, 415, 297, 597, 368, 31, 278}
        initial_budget = 35
    elif season == '2024-25':
        initial_squad = {91, 455, 335, 291, 285, 328, 182, 348, 442, 401, 351, 414, 276, 475, 583}
        initial_budget = 0

    return initial_squad, initial_budget


def run_simulation(season: str, wildcard_gameweeks=[14, 25], log=False, use_cache=True) -> int:
    """Simulate a season of FPL and return the overall points scored."""
    
    # Load season results (total_points and minutes for each player in each round)
    results = load_simulation_true_results(season, use_cache=use_cache)
    total_points = results['total_points']
    minutes = results['minutes']

    # Initialize season statistics
    first_gameweek = 1
    last_gameweek = 38
    overall_points = 0
    free_transfers = 0
    current_squad, current_budget = get_initial_team_and_budget(season)
    purchase_prices = load_simulation_purchase_prices(season, current_squad, first_gameweek)


    for next_gameweek in range(first_gameweek, last_gameweek + 1):

        if (season == '2022-23') and (next_gameweek == 7):
            continue

        # Load data for the next gameweek
        elements = load_simulation_bootstrap_elements(season, next_gameweek)
        positions = elements['element_type']
        now_costs = elements['now_cost']
        selling_prices = get_selling_prices(current_squad, purchase_prices, now_costs)

        # Make predictions
        features = load_simulation_features(season, next_gameweek, use_cache=use_cache)
        model_path = f"models/2024-25/model-excluded-{season}.pkl"
        columns_path = f"models/2024-25/columns.json"
        predictions = make_predictions(features, model_path, columns_path)
        predictions = group_predictions_by_gameweek(predictions)
        predictions = weight_gameweek_predictions_by_availability(predictions, elements, next_gameweek)

        # Optimize the squad for the next gameweek
        solution = optimize_squad(
            season,
            current_squad,
            current_budget,
            free_transfers,
            next_gameweek,
            wildcard_gameweeks,
            predictions,
            elements,
            now_costs,
            selling_prices
        )
        best_squad = solution['squad']
        best_roles = {
            'starting_xi': solution['starting_xi'],
            'captain': solution['captain'],
            'vice_captain': solution['vice_captain'],
            'reserve_gkp': solution['reserve_gkp'],
            'reserve_out': solution['reserve_out'],
        }

        # Update budget, purchase prices, and selling prices
        updated_budget = calculate_budget(
            current_squad, best_squad, current_budget, selling_prices, now_costs
        )
        updated_purchase_prices = update_purchase_prices(
            purchase_prices, now_costs, current_squad, best_squad
        )
        updated_selling_prices = update_selling_prices(
            selling_prices, now_costs, current_squad, best_squad
        )

        # Perform automatic substitutions
        automatic_roles = make_automatic_substitutions(
            best_roles, minutes.loc[:, next_gameweek], positions
        )

        # Calculate the number of points scored
        gameweek_points = calculate_points(
            roles=automatic_roles,
            total_points=total_points.loc[:, next_gameweek],
            captain_multiplier=2,
            starting_xi_multiplier=1,
            reserve_gkp_multiplier=0,
            reserve_out_multiplier=0
        )

        # Calculate the cost of transfers
        if next_gameweek not in {1, *wildcard_gameweeks}:
            transfers_made = count_transfers_made(best_squad, current_squad)
        else:
            transfers_made = 0

        transfer_cost = calculate_transfer_cost(free_transfers, transfers_made)

        # Update the overall points
        overall_points += gameweek_points 
        overall_points -= transfer_cost

        if log:
            print_gameweek_report(
                next_gameweek,
                current_squad,
                best_squad,
                best_roles,
                automatic_roles,
                updated_budget,
                selling_prices,
                updated_purchase_prices,
                total_points.loc[:, next_gameweek],
                positions,
                elements,
                gameweek_points,
                free_transfers,
                transfer_cost,
                overall_points
            )

        # Update the squad, budget, prices, and free transfers
        current_squad = best_squad
        current_budget = updated_budget
        purchase_prices = updated_purchase_prices
        selling_prices = updated_selling_prices

        if next_gameweek in wildcard_gameweeks:
            # Free transfers are rolled over after playing a wildcard
            pass
        else:
            free_transfers = max(free_transfers - transfers_made + 1, 1)
            free_transfers = min(free_transfers, MAX_FREE_TRANSFERS)
            
    return overall_points