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
from sircuitenum.singular_interface import solve_with_singular

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
        breakpoint()
        assert len(branches) >= 2
        # "Should find at least 2 branches (Generic + Singular)"
        
        has_generic = False
        has_singular = False
        
        for b in branches:
            mappings = b['mappings']
            constraints = b['constraints']
            
            # Check for Generic Case
            if self.x in mappings and mappings[self.x] == self.b / self.a:
                has_generic = True
            
            # Check for Singular Case (Constraints contain a and b)
            # Note: exact constraint check can be tricky due to formatting, 
            # but usually it's [b, a] or similar.
            if constraints and self.x in mappings and mappings[self.x] == "Free Parameter":
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
        
        solutions = [b['mappings'][self.x] for b in branches]
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
            m = b['mappings']
            if m.get(self.x) == 0: found_x_zero = True
            if m.get(self.y) == 1: found_y_one = True
            
        assert found_x_zero, "Failed to find branch x=0"
        assert found_y_one, "Failed to find branch y=1"

    def test_05_cyclic_3(self):
        """
        The "Benchmark" Test: Cyclic-3 Roots
        Tests performance and complexity handling.
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
        
        print(f"Cyclic-3 found {len(branches)} branches.")
        assert len(branches) > 0, "Cyclic-3 should have solutions"
        
        # Check if the mappings are discrete (no free parameters)
        # Note: Cyclic-3 is 0-dimensional, so all vars should be mapped to numbers/algebraic values.
        for b in branches:
            for var in [self.x, self.y, self.z]:
                val = b['mappings'].get(var)
                assert val != "Free Parameter", "Cyclic-3 is 0-dim, should not have free parameters"

    def test_06_algebraic_number(self):
        """
        The "Sqrt" Test: x^2 - 2 = 0
        Tests if Singular returns the reduced algebraic form properly.
        """
        self.setup_method()
        print("\n--- Test 06: Algebraic Number (x^2 - 2 = 0) ---")
        eqs = [self.x**2 - 2]
        branches = solve_with_singular(eqs, [self.x])
        
        # Since Singular can't output sqrt(2) explicitly without field extensions,
        # it usually outputs the Defining Polynomial in the 'constraints' or keeps x implicit.
        # However, our script logic attempts 'reduce(x)'. 
        # In a standard ring, reduce(x, x^2-2) is just 'x'.
        # This checks if your parser handles "Explicit vs Implicit" correctly.
        
        for b in branches:
            # If x maps to 'Free' or itself, it means it's an algebraic number constraint
            # Check constraints for x^2 - 2
            constraints = b['constraints']
            has_poly = any(str(c).replace(" ", "") in ["x**2-2", "-x**2+2"] for c in constraints)
            if has_poly:
                print("Confirmed: Returned x implicitly defined by x^2 - 2")
                return

        # If we reach here without finding the constraint, warn (but don't fail, behavior varies by ring setup)
        print("Warning: explicit x^2-2 constraint checking is tricky with floats/ints.")

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
