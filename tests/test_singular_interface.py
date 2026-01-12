"""
Tests for singular_interface module.

Tests the Groebner Cover functionality using examples from the
Singular grobcov.lib documentation.

Author: Eli Weissler
"""

import pytest
import sympy
from sympy import symbols, sympify

# IMPORT YOUR SOLVER HERE
from sircuitenum.singular_interface import solve_with_singular, parse_singular_output 

class TestSingularParser():
   
    def test_parse(self):
        
        raw_output = """
        |||START||||||BRANCH|||1
        |||COMPONENT|||1
        |||PARAMS|||a,b
        |||VARS|||x,y
        |||CONSTRAINTS|||
        |||NUM_CONSTRAINT_SOLUTIONS|||-1
        |||NONNULL|||
        (a)
        |||BASIS|||
        (a2)*y+(-a3+b2)
        (a)*x+(-b)
        |||NUM_SOLUTIONS|||1
        |||BRANCH|||2
        |||COMPONENT|||1
        |||PARAMS|||a,b
        |||VARS|||x,y
        |||CONSTRAINTS|||
        (a)
        |||NUM_CONSTRAINT_SOLUTIONS|||1
        |||CONSTRAINT_BASIS|||
        a
        |||NONNULL|||
        (b),(a)
        |||BASIS|||
        1
        |||NUM_SOLUTIONS|||0
        |||BRANCH|||3
        |||COMPONENT|||1
        |||PARAMS|||a,b
        |||VARS|||x,y
        |||CONSTRAINTS|||
        (b)
        (a)
        |||NUM_CONSTRAINT_SOLUTIONS|||1
        |||CONSTRAINT_BASIS|||
        b
        a
        |||NONNULL|||
        1
        |||BASIS|||
        |||NUM_SOLUTIONS|||-1
        |||END|||
        """
        branches = parse_singular_output(raw_output, potential_vars=['x', 'y'], fixed_params=['a', 'b'])
        assert len(branches) == 2 # One branch has zero solutions
        assert branches[0]['vars'] == ['x', 'y']
        assert branches[0]['num_solutions'] == 1
        assert [str(x) for x in branches[1]['constraints']] == ['b', 'a']

class TestSingularSolver():

    def setup_method(self):
        # Define commonly used symbols
        self.a, self.b = symbols('a b')  # Parameters
        self.x, self.y, self.z = symbols('x y z')  # Unknowns

    def test_01_parametric_singularity(self):
        """
        The "Divider" Test: ax = b
        Tests if the solver detects that 'a' cannot be zero without a special branch.
        """
        self.setup_method()
        print("\n--- Test 01: Parametric Singularity (ax = b) ---")
        eqs = [self.a * self.x - self.b]
        branches = solve_with_singular(eqs, [self.x])
        
        # Expectation: 
        # Branch 1: Generic (x = b/a)
        # Branch 2: Singular (a=0, b=0, x=Free)
        assert len(branches) == 2
        # "Should find at least 2 branches (Generic + Singular)"
        
        has_generic = False
        has_singular = False
        
        for b in branches:
            mappings = b['mappings']
            constraints = b['constraints']
            
            # Check for Generic Case
            if self.x in mappings[0] and mappings[0][self.x] == self.b / self.a:
                has_generic = True
            
            # Check for Singular Case (Constraints contain a and b)
            # Note: exact constraint check can be tricky due to formatting, 
            # but usually it's [b, a] or similar.
            if constraints and self.a in mappings[0] and self.b in mappings[0] and self.x in b["free_vars"]:
                has_singular = True
        
        # Failed to find generic solution x = b/a
        assert has_generic
        # Failed to find singular case where x is Free
        assert has_singular

    def test_02_reducible_geometry(self):
        """
        The "Crossing" Test: x^2 - y^2 = 0
        Tests if facstd correctly splits the geometry into two linear branches.
        """
        self.setup_method()
        print("\n--- Test 02: Reducible Geometry (x^2 - y^2 = 0) ---")
        eqs = [self.x**2 - self.y**2]
        branches = solve_with_singular(eqs, [self.x, self.y])
        
        # Expectation:
        # Branch 1: x = y
        # Branch 2: x = -y (or vice versa)
        assert len(branches) == 2, "Should split x^2-y^2 into exactly 2 linear branches"
        
        solutions = [b['mappings'][0][self.x] for b in branches]
        assert self.y in solutions
        assert -self.y in solutions

    def test_03_inconsistent_system(self):
        """
        The "Impossible" Test: 1 = 0
        Tests if the solver correctly returns an empty list for no solutions.
        """
        print("\n--- Test 03: Inconsistent System (1 = 0) ---")
        self.setup_method()
        eqs = [sympify(1)] # 1 = 0
        branches = solve_with_singular(eqs, [self.x])
        
        assert len(branches) == 0, "Should return 0 branches for inconsistent system"

    def test_04_mixed_dimension(self):
        """
        The "Mixed Dim" Test: x(y-1) = 0
        Tests if it can return a Plane (dim 2) and a Line (dim 1) simultaneously.
        """
        print("\n--- Test 04: Mixed Dimension (x(y-1) = 0) ---")
        self.setup_method()
        eqs = [self.x * (self.y - 1)]
        branches = solve_with_singular(eqs, [self.x, self.y])
        
        # Expectation:
        # Case A: x = 0 (y is Free) -> Plane
        # Case B: y = 1 (x is Free) -> Plane/Line depending on z (here just x,y so lines)
        
        found_x_zero = False
        found_y_one = False
        
        for b in branches:
            m = b['mappings'][0]
            if m.get(self.x) == 0: found_x_zero = True
            if m.get(self.y) == 1: found_y_one = True
            
        assert found_x_zero, "Failed to find branch x=0"
        assert found_y_one, "Failed to find branch y=1"

    def test_05_cyclic_3(self):
        """
        x + y + z = 0
        xy + yz + zx = 0
        xyz - 1 = 0
        """
        print("\n--- Test 05: Cyclic-3 Benchmark ---")
        self.setup_method()
        eqs = [
            self.x + self.y + self.z,
            self.x*self.y + self.y*self.z + self.z*self.x,
            self.x*self.y*self.z - 1
        ]
        # This system has exactly 6 discrete solutions (permutations of roots of unity)
        branches = solve_with_singular(eqs, [self.x, self.y, self.z])
        
        assert len(branches) == 3, "Cyclic-3 should have 3 solutions"
        
        # Check if the mappings are discrete (no free parameters)
        # Note: Cyclic-3 is 0-dimensional, so all vars should be mapped to numbers/algebraic values.
        for b in branches:
            assert b["free_vars"] == [], "Cyclic-3 solutions should have no free variables"
            assert b["vars"] == ['x', 'y', 'z'], "Cyclic-3 should have x,y,z as variables"


    def test_06_algebraic_number(self):
        """
        Tests if Singular returns the reduced algebraic form properly.
        """
        self.setup_method()
        print("\n--- Test 06: Algebraic Number (x^2 - 2 = 0) ---")
        eqs = [self.x**2 - 2]
        branches = solve_with_singular(eqs, [self.x])
        assert len(branches) == 1
        assert branches[0]["num_solutions"] == 2, "Should have 2 solutions for x^2 - 2 = 0"
        for var, val in branches[0]['mappings'][0].items():
            assert var == self.x
            assert val == sympy.sqrt(2) or val == -sympy.sqrt(2), "Algebraic number solution incorrect"

if __name__ == "__main__":
    # Run tests
    # pytest.main([__file__, "-v"])
    test_solver = TestSingularSolver()
    test_solver.test_01_parametric_singularity()
    test_solver.test_02_reducible_geometry()
    test_solver.test_03_inconsistent_system()
    test_solver.test_04_mixed_dimension()
    test_solver.test_05_cyclic_3()
    test_solver.test_06_algebraic_number()
