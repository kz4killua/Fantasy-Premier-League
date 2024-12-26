import uuid

import numpy as np
from ortools.linear_solver import pywraplp


GKP = 1
DEF = 2
MID = 3
FWD = 4


def solve(
    gameweeks: list[int],
    players: list[int],
    teams: dict[int, int],
    element_types: dict[int, int],
    points: dict[int, float],
    initial_squad: set[int],
    initial_budget: int,
    initial_free_transfers: int,
    purchase_prices: dict[int, int],
    selling_prices: dict[int, int],
    wildcards: list[int],

    # Optimization parameters
    squad_evaluation_round_factor: float,
    starting_xi_multiplier: float,
    captain_multiplier: float,
    vice_captain_multiplier: float,
    reserve_gkp_multiplier: float,
    reserve_1_multiplier: float,
    reserve_2_multiplier: float,
    reserve_3_multiplier: float,

    log: bool = False,
):
    
    solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
    if not solver:
        return
    
    variables = create_variables(
        solver, 
        players, 
        gameweeks
    )
    create_constraints(
        solver, 
        variables, 
        initial_squad, 
        initial_budget, 
        initial_free_transfers,
        gameweeks,
        wildcards, 
        purchase_prices,
        selling_prices,
        players, 
        teams, 
        element_types
    )
    create_objective(
        solver, 
        variables, 
        players, 
        gameweeks, 
        points,
        starting_xi_multiplier,
        captain_multiplier,
        vice_captain_multiplier,
        reserve_gkp_multiplier,
        reserve_1_multiplier,
        reserve_2_multiplier,
        reserve_3_multiplier,
        squad_evaluation_round_factor,
    )

    status = solver.Solve()

    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        if log:
            print(f"Solved in {solver.wall_time():d} milliseconds")
            print(f"Objective value = {solver.Objective().Value()}")
            print(f"Optimal = {status == pywraplp.Solver.OPTIMAL}")
        return get_solution(variables, gameweeks)
    else:
        raise Exception("No solution found.")


def get_solution(
    variables: dict,
    gameweeks: list[int],
):

    solutions = {}

    for gameweek in gameweeks:

        squad = [p for (p, g), v in variables['squad'].items() if g == gameweek and v.solution_value() == 1]
        starting_xi = [p for (p, g), v in variables['starting_xi'].items() if g == gameweek and v.solution_value() == 1]
        captain = [p for (p, g), v in variables['captain'].items() if g == gameweek and v.solution_value() == 1]
        vice_captain = [p for (p, g), v in variables['vice_captain'].items() if g == gameweek and v.solution_value() == 1]
        reserve_gkp = [p for (p, g), v in variables['reserve_gkp'].items() if g == gameweek and v.solution_value() == 1]
        reserve_1 = [p for (p, g), v in variables['reserve_1'].items() if g == gameweek and v.solution_value() == 1]
        reserve_2 = [p for (p, g), v in variables['reserve_2'].items() if g == gameweek and v.solution_value() == 1]
        reserve_3 = [p for (p, g), v in variables['reserve_3'].items() if g == gameweek and v.solution_value() == 1]
        
        purchases = [p for (p, g), v in variables['purchases'].items() if g == gameweek and v.solution_value() == 1]
        sales = [p for (p, g), v in variables['sales'].items() if g == gameweek and v.solution_value() == 1]
        budget = [v.solution_value() for g, v in variables['budget'].items() if g == gameweek][0]
        free_transfers = [v.solution_value() for g, v in variables['free_transfers'].items() if g == gameweek][0]
        paid_transfers = [v.solution_value() for g, v in variables['paid_transfers'].items() if g == gameweek][0]

        # Sanity checks: Ensure each selection is valid
        assert len(squad) == 15
        assert len(starting_xi) == 11
        assert len(captain) == 1
        assert len(vice_captain) == 1
        assert len(reserve_gkp) == 1
        assert len(reserve_1) == 1
        assert len(reserve_2) == 1
        assert len(reserve_3) == 1
        assert set(squad) == set(starting_xi + reserve_gkp + reserve_1 + reserve_2 + reserve_3)
        assert len(purchases) == len(sales)

        solutions[gameweek] = {
            'squad': set(squad),
            'starting_xi': starting_xi,
            'captain': captain[0],
            'vice_captain': vice_captain[0],
            'reserve_gkp': reserve_gkp[0],
            'reserve_out': [reserve_1[0], reserve_2[0], reserve_3[0]],
            'purchases': purchases,
            'sales': sales,
            'budget': budget,
            'free_transfers': free_transfers,
            'paid_transfers': paid_transfers,
        }

    return solutions


def create_variables(
    solver: pywraplp.Solver,
    players: list[int],
    gameweeks: list[int],
):
    variables = dict()

    # Variables for squad selections
    variables['squad'] = {(p, g): solver.IntVar(0, 1, f'squad_{p}_{g}') for p in players for g in gameweeks}
    variables['captain'] = {(p, g): solver.IntVar(0, 1, f'captain_{p}_{g}') for p in players for g in gameweeks}
    variables['vice_captain'] = {(p, g): solver.IntVar(0, 1, f'vice_captain_{p}_{g}') for p in players for g in gameweeks}
    variables['starting_xi'] = {(p, g): solver.IntVar(0, 1, f'starting_xi_{p}_{g}') for p in players for g in gameweeks}
    variables['reserve_gkp'] = {(p, g): solver.IntVar(0, 1, f'reserve_gkp_{p}_{g}') for p in players for g in gameweeks}
    variables['reserve_1'] = {(p, g): solver.IntVar(0, 1, f'reserve_1_{p}_{g}') for p in players for g in gameweeks}
    variables['reserve_2'] = {(p, g): solver.IntVar(0, 1, f'reserve_2_{p}_{g}') for p in players for g in gameweeks}
    variables['reserve_3'] = {(p, g): solver.IntVar(0, 1, f'reserve_3_{p}_{g}') for p in players for g in gameweeks}

    # Variables for transfers
    variables['budget'] = {g: solver.IntVar(0, solver.infinity(), f'budget_{g}') for g in gameweeks}
    variables['free_transfers'] = {g: solver.IntVar(0, 15, f'free_transfers_{g}') for g in gameweeks}
    variables['paid_transfers'] = {g: solver.IntVar(0, 15, f'paid_transfers_{g}') for g in gameweeks}
    variables['purchases'] = {(p, g): solver.IntVar(0, 1, f'purchase_{p}_{g}') for p in players for g in gameweeks}
    variables['sales'] = {(p, g): solver.IntVar(0, 1, f'sale_{p}_{g}') for p in players for g in gameweeks}

    # Variables for chips
    variables['wildcards'] = {g: solver.IntVar(0, 1, f'wildcard_{g}') for g in gameweeks}

    return variables


def create_constraints(
    solver: pywraplp.Solver,
    variables: dict,
    initial_squad: set[int],
    initial_budget: int,
    initial_free_transfers: int,
    gameweeks: list[int],
    wildcards: list[int],
    purchase_prices: dict[int, int],
    selling_prices: dict[int, int],
    players: list[int],
    teams: dict[int, int],
    element_types: dict[int, int],
):

    # There must be no more than 3 players from the same team
    for g in gameweeks:
        for t in set(teams.values()):
            solver.Add(
                sum(variables['squad'][p, g] for p in players if teams[p] == t) <= 3
            )

    # There must be exactly 2 goalkeepers, 5 defenders, 5 midfielders, and 3 forwards (i.e. 15 players)
    for g in gameweeks:
        solver.Add(sum(variables['squad'][p, g] for p in players if element_types[p] == GKP) == 2)
        solver.Add(sum(variables['squad'][p, g] for p in players if element_types[p] == DEF) == 5)
        solver.Add(sum(variables['squad'][p, g] for p in players if element_types[p] == MID) == 5)
        solver.Add(sum(variables['squad'][p, g] for p in players if element_types[p] == FWD) == 3)

    # There must be exactly 1 captain, 1 vice captain, etc.
    for g in gameweeks:
        solver.Add(sum(variables['captain'][p, g] for p in players) == 1)
        solver.Add(sum(variables['vice_captain'][p, g] for p in players) == 1)
        solver.Add(sum(variables['starting_xi'][p, g] for p in players) == 11)
        solver.Add(sum(variables['reserve_gkp'][p, g] for p in players) == 1)
        solver.Add(sum(variables['reserve_1'][p, g] for p in players) == 1)
        solver.Add(sum(variables['reserve_2'][p, g] for p in players) == 1)
        solver.Add(sum(variables['reserve_3'][p, g] for p in players) == 1)

    # The reserve goalkeeper must be a GKP
    for g in gameweeks:
        solver.Add(
            sum(variables['reserve_gkp'][p, g] for p in players if element_types[p] == GKP) == 1
        )

    # There must be exactly 1 starting GKP, at least 3 starting DEFs, and at least 1 starting FWD
    for g in gameweeks:
        solver.Add(sum(variables['starting_xi'][p, g] for p in players if element_types[p] == GKP) == 1)
        solver.Add(sum(variables['starting_xi'][p, g] for p in players if element_types[p] == DEF) >= 3)
        solver.Add(sum(variables['starting_xi'][p, g] for p in players if element_types[p] == FWD) >= 1)

    # The starting XI and reserve players must be a subset of the squad
    for g in gameweeks:
        for p in players:
            solver.Add(variables['starting_xi'][p, g] <= variables['squad'][p, g])
            solver.Add(variables['reserve_gkp'][p, g] <= variables['squad'][p, g])
            solver.Add(variables['reserve_1'][p, g] <= variables['squad'][p, g])
            solver.Add(variables['reserve_2'][p, g] <= variables['squad'][p, g])
            solver.Add(variables['reserve_3'][p, g] <= variables['squad'][p, g])

    # The captain and vice captain must be in the starting XI (and hence in the squad)
    for g in gameweeks:
        for p in players:
            solver.Add(variables['captain'][p, g] <= variables['starting_xi'][p, g])
            solver.Add(variables['vice_captain'][p, g] <= variables['starting_xi'][p, g])

    # The reserve players must not be in the starting XI
    for g in gameweeks:
        for p in players:
            solver.Add(variables['reserve_gkp'][p, g] + variables['starting_xi'][p, g] <= 1)
            solver.Add(variables['reserve_1'][p, g] + variables['starting_xi'][p, g] <= 1)
            solver.Add(variables['reserve_2'][p, g] + variables['starting_xi'][p, g] <= 1)
            solver.Add(variables['reserve_3'][p, g] + variables['starting_xi'][p, g] <= 1)

    # The captain, vice captain, and reserve players must be disjoint
    for g in gameweeks:
        for p in players:
            solver.Add(
                variables['captain'][p, g] 
                + variables['vice_captain'][p, g] 
                + variables['reserve_gkp'][p, g] 
                + variables['reserve_1'][p, g] 
                + variables['reserve_2'][p, g] 
                + variables['reserve_3'][p, g] 
                <= 1
            )

    # The budget must be non-negative
    for g in gameweeks:
        solver.Add(variables['budget'][g] >= 0)

    # The budget must be consistent with purchases and sales
    for i, g in enumerate(gameweeks):

        if i == 0:
            prior_budget = initial_budget
        else:
            prior_budget = variables['budget'][gameweeks[i - 1]]

        income = sum(variables['sales'][p, g] * selling_prices[p] for p in players)
        expenses = sum(variables['purchases'][p, g] * purchase_prices[p] for p in players)

        solver.Add(variables['budget'][g] == prior_budget + income - expenses)

    # The squad must be consistent with purchases and sales
    for i, g in enumerate(gameweeks):
        for p in players:

            if i == 0:
                in_prior_squad = int(p in initial_squad)
            else:
                in_prior_squad = variables['squad'][p, gameweeks[i - 1]]

            # Purchases add players to the squad, sales remove them
            solver.Add(
                variables['squad'][p, g] == (
                    in_prior_squad 
                    + variables['purchases'][p, g] 
                    - variables['sales'][p, g]
                )
            )

            # Sell only owned players and buy only unowned players
            solver.Add(
                variables['sales'][p, g] <= in_prior_squad
            )
            solver.Add(
                variables['purchases'][p, g] <= 1 - in_prior_squad
            )

    # Wildcards are not available in gameweek 1
    if 1 in wildcards:
        raise ValueError("Wildcards are not available in gameweek 1.")

    # Activate wildcards in selected gameweeks
    for g in gameweeks:
        solver.Add(variables['wildcards'][g] == int(g in wildcards))

    # Update the number of paid transfers
    for g in gameweeks:

        # In gameweek 1 and on wildcard gameweeks, no transfers are paid
        if g in {1, *wildcards}:
            solver.Add(variables['paid_transfers'][g] == 0)

        # In other weeks, paid transfers = max(transfers made - free transfers, 0)
        else:
            paid_transfers = variables['paid_transfers'][g]
            free_transfers = variables['free_transfers'][g]
            transfers_made = sum(variables['sales'][p, g] for p in players)
            _create_max_constraint(solver, paid_transfers, transfers_made - free_transfers, 0, 15)

    # Update the number of free transfers
    for i, g in enumerate(gameweeks):

        if i == 0:
            solver.Add(variables['free_transfers'][g] == initial_free_transfers)

        # In gameweek 1, there are no free transfers
        elif g == 1:
            solver.Add(variables['free_transfers'][g] == 0)

        # Free transfers are rolled over after playing a wildcard
        elif g - 1 in wildcards:
            solver.Add(variables['free_transfers'][g] == variables['free_transfers'][gameweeks[i - 1]])
        
        # In the general case:
        # FT2 = min(max(1, FT1 - TM1 + 1), 2)
        # where: 
        # - TM1 is the number of transfers made in the previous gameweek
        # - FT1 is the number of free transfers in the previous gameweek
        # - FT2 is the number of free transfers in the current gameweek
        # Inspired by @sertalpbilal: https://youtu.be/Prv8M7hE3vk?t=2079
        else:
            ft2 = variables['free_transfers'][g]
            ft1 = variables['free_transfers'][gameweeks[i - 1]]
            tm1 = sum(variables['sales'][p, gameweeks[i - 1]] for p in players)

            # Enforce the inner max constraint
            inner = solver.IntVar(1, 15, f'_{uuid.uuid4()}')
            _create_max_constraint(solver, inner, 1, ft1 - tm1 + 1, 15)

            # The outer min constraint limits the number of free transfers
            _create_min_constraint(solver, ft2, inner, 2, 15)


def create_objective(
    solver: pywraplp.Solver,
    variables: dict,
    players: list[int],
    gameweeks: list[int],
    points: dict[int, float],

    # Optimization parameters
    starting_xi_multiplier: float,
    captain_multiplier: float,
    vice_captain_multiplier: float,
    reserve_gkp_multiplier: float,
    reserve_1_multiplier: float,
    reserve_2_multiplier: float,
    reserve_3_multiplier: float,
    squad_evaluation_round_factor: float,
):
    
    scores = []

    for g in gameweeks:
        
        score = 0

        # Sum the points for all players in the starting XI
        score += sum(
            variables['starting_xi'][p, g] * points[p, g] * starting_xi_multiplier for p in players
        )

        # Account for the captain and vice captain multipliers (as they are part of the starting XI)
        score += sum(
            variables['captain'][p, g] * points[p, g] * (captain_multiplier - starting_xi_multiplier) for p in players
        )
        score += sum(
            variables['vice_captain'][p, g] * points[p, g] * (vice_captain_multiplier - starting_xi_multiplier) for p in players
        )

        # Sum the points for all players in the reserve
        score += sum(
            variables['reserve_gkp'][p, g] * points[p, g] * reserve_gkp_multiplier for p in players
        )
        score += sum(
            variables['reserve_1'][p, g] * points[p, g] * reserve_1_multiplier for p in players
        )
        score += sum(
            variables['reserve_2'][p, g] * points[p, g] * reserve_2_multiplier for p in players
        )
        score += sum(
            variables['reserve_3'][p, g] * points[p, g] * reserve_3_multiplier for p in players
        )

        # Add transfer costs
        score -= variables['paid_transfers'][g] * 4

        scores.append(score)


    # Calculate the decay-weighted sum of gameweek scores
    objective = calculate_weighted_sum(scores, squad_evaluation_round_factor)

    solver.Maximize(objective)


# TODO: Share this function with optimize/utilities.py
def calculate_weighted_sum(
    scores: list[float] | list[pywraplp.Variable],
    decay: float,
):
    weights = decay ** np.arange(len(scores))
    weights /= weights.sum()
    return sum(s * w for s, w in zip(scores, weights))


def _create_max_constraint(
    solver: pywraplp.Solver,
    y: pywraplp.Variable | pywraplp.LinearExpr,
    a: pywraplp.Variable | pywraplp.LinearExpr,
    b: pywraplp.Variable | pywraplp.LinearExpr,
    m: int,
    z: pywraplp.Variable | pywraplp.LinearExpr | None = None,
):
    """
    Enforce the constraint y = max(a, b).

    This constraint is equivalent to the following linear inequalities:
    - y >= a
    - y >= b
    - y <= a + m * (1 - z)
    - y <= b + m * z

    where: 
    - m is a sufficiently large constant such that a, b <= m for any "reasonable" solution
    - z is a binary variable such that z = 1 if a >= b, and z = 0 otherwise

    If z is not provided, it is initialized as a binary variable.

    See: https://or.stackexchange.com/questions/711/how-to-formulate-linearize-a-maximum-function-in-a-constraint
    """

    # Create the binary variable z if not provided
    if z is None:
        z = solver.IntVar(0, 1, f'_{uuid.uuid4()}')

    # Add the linear inequalities
    solver.Add(y >= a)
    solver.Add(y >= b)
    solver.Add(y <= a + m * (1 - z))
    solver.Add(y <= b + m * z)


def _create_min_constraint(
    solver: pywraplp.Solver,
    y: pywraplp.Variable | pywraplp.LinearExpr,
    a: pywraplp.Variable | pywraplp.LinearExpr,
    b: pywraplp.Variable | pywraplp.LinearExpr,
    m: int,
    z: pywraplp.Variable | pywraplp.LinearExpr | None = None,
):
    """
    Enforce the constraint y = min(a, b).

    This constraint is equivalent to the following linear inequalities:
    - y <= a
    - y <= b
    - y >= a - m * (1 - z)
    - y >= b - m * z

    where: 
    - m is a sufficiently large constant such that a, b <= m for any "reasonable" solution
    - z is a binary variable such that z = 1 if a <= b, and z = 0 otherwise

    If z is not provided, it is initialized as a binary variable.

    See: https://or.stackexchange.com/questions/1160/how-to-linearize-min-function-as-a-constraint
    """

    # Create the binary variable z if not provided
    if z is None:
        z = solver.IntVar(0, 1, f'_{uuid.uuid4()}')

    # Add the linear inequalities
    solver.Add(y <= a)
    solver.Add(y <= b)
    solver.Add(y >= a - m * (1 - z))
    solver.Add(y >= b - m * z)
