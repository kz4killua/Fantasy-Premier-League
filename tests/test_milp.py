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
        y1 = solver.IntVar(0, 10, 'y1')
        y2 = solver.IntVar(0, 10, 'y2')
        y3 = solver.IntVar(0, 10, 'y3')

        solver.Add(a == 5)
        solver.Add(b == 9)

        create_max_constraint(solver, y1, a, b, 100)
        create_max_constraint(solver, y2, a + 3, b, 100)
        create_max_constraint(solver, y3, a + 5, b, 100)

        solver.Solve()

        self.assertEqual(y1.solution_value(), 9)
        self.assertEqual(y2.solution_value(), 9)
        self.assertEqual(y3.solution_value(), 10)
        

    def test_create_min_constraint(self):

        solver = pywraplp.Solver.CreateSolver("CBC_MIXED_INTEGER_PROGRAMMING")
        if not solver:
            return

        a = solver.IntVar(0, 10, 'a')
        b = solver.IntVar(0, 10, 'b')
        y1 = solver.IntVar(0, 10, 'y1')
        y2 = solver.IntVar(0, 10, 'y2')
        y3 = solver.IntVar(0, 10, 'y3')

        solver.Add(a == 5)
        solver.Add(b == 9)

        create_min_constraint(solver, y1, a, b, 100)
        create_min_constraint(solver, y2, a + 3, b - 5, 100)
        create_min_constraint(solver, y3, a + 5, b, 100)

        solver.Solve()

        self.assertEqual(y1.solution_value(), 5)
        self.assertEqual(y2.solution_value(), 4)
        self.assertEqual(y3.solution_value(), 9)