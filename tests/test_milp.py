import unittest

from optimize.milp import create_max_constraint, create_min_constraint
from ortools.linear_solver import pywraplp


class TestMILP(unittest.TestCase):

    def test_create_max_constraint(self):

        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if not solver:
            return

        a = solver.IntVar(0, 10, 'a')
        b = solver.IntVar(0, 10, 'b')
        y = solver.IntVar(0, 10, 'y')

        solver.Add(a == 5)
        solver.Add(b == 9)

        create_max_constraint(solver, y, a, b, 100)

        solver.Solve()

        self.assertEqual(y.solution_value(), 9)
        

    def test_create_min_constraint(self):

        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if not solver:
            return

        a = solver.IntVar(0, 10, 'a')
        b = solver.IntVar(0, 10, 'b')
        y = solver.IntVar(0, 10, 'y')

        solver.Add(a == 5)
        solver.Add(b == 9)

        create_min_constraint(solver, y, a, b, 100)

        solver.Solve()

        self.assertEqual(y.solution_value(), 5)