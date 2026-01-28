"""
Tests for singular_interface module.

Tests the Groebner Cover functionality using examples from the
Singular grobcov.lib documentation.

Author: Eli Weissler
"""


import pytest
import psutil
import time
import os
import re
from pathlib import Path
import sympy as sym
from sympy import symbols, sympify


from sircuitenum.singular_interface import solve_with_singular, parse_singular_output, filter_redundant_branches
from sircuitenum.singular_interface import extract_mappings, _robust_substitute, is_compatible, _eq_as_numer_denom
from sircuitenum.singular_interface import SafeSingular, solve_0D_backsub, get_sage_groebner_basis


import json
import json
import sympy as sym

import json
import sympy as sym

from sage.all import singular, PolynomialRing, QQ, ideal, SR, var, QQbar


def load_mathematica_benchmark(json_path, system_vars):
    """
    Loads JSON from Mathematica Solve and converts to standard branch format.
    Robustly handles Mathematica syntax ([], &&, ^) by converting to Python syntax.
    """
    with open(json_path, 'r') as f:
        raw_data = json.load(f)
        
    branches = []
    
    def clean_syntax(s):
        s = re.sub(r'\(-1\)\^\(([^)]+)\)', r'exp(I*pi*(\1))', s)
        return s.replace('^', '**').replace('Sqrt', 'sqrt').replace('[', '(').replace(']', ')')

    for i, item in enumerate(raw_data):
        mapping = {}
        nonnull = []
        constraints_list = [] # NEW: Stores full SymPy objects
        
        # 1. Process Constraints
        for c in item.get('constraints', []):
            try:
                clean_c = clean_syntax(c)
                
                # Robust parsing for "!=" string
                if "!=" in clean_c:
                    lhs_str, rhs_str = clean_c.split("!=")
                    lhs = sym.sympify(lhs_str)
                    rhs = sym.sympify(rhs_str)
                    
                    # Create SymPy Inequality Object
                    constr_obj = sym.Ne(lhs, rhs)
                    
                    # Add difference to nonnull list
                    nonnull.append(lhs - rhs)
                else:
                    # standard expressions
                    expr = sym.sympify(clean_c)
                    constraints_list.append(expr)
                    
                    if isinstance(expr, sym.Ne):
                        nonnull.append(expr.args[0] - expr.args[1])
                    else:
                        nonnull.append(expr)
                        
            except Exception as e:
                print(f"Warning: Could not parse constraint '{c}': {e}")
                pass 

        # 2. Process Mappings
        for k, v in item['mapping'].items():
            if v == "Undefined":
                mapping = None
                break 
            
            key = sym.sympify(k)
            val = sym.sympify(clean_syntax(v))
            mapping[key] = val
        
        if mapping is None:
            continue
            
        free_params = [v for v in system_vars if v not in mapping]
        
        branches.append({
            'id': f"{i+1}",
            'basis': [], 
            'mapping': mapping,
            'constraints': list(set(constraints_list)),
            'nonnull': list(set(nonnull)),
            'params': free_params
        })
        
    return branches


# --- 2. PARSER FOR MATHEMATICA REDUCE JSON ---
def parse_mathematica_reduce_json(json_path, sys_vars):
    local_dict = {str(v): v for v in sys_vars}
    local_dict['I'] = sym.I
    local_dict['sqrt'] = sym.sqrt
    
    def math_to_py_syntax(expr_str):
        s = expr_str.replace("^", "**")
        s = s.replace("Sqrt[", "sqrt(").replace("]", ")")
        return s

    with open(json_path, 'r') as f:
        raw_data = json.load(f)

    clean_branches = []
    for item in raw_data:
        branch = {
            'id': f"Reduce_{item['id']}",
            'basis': [],
            'mapping': {},
            'nonnull': [],
            'params': [],
            'constraints': [] 
        }

        # Equalities (==)
        for eq_str in item['equalities']:
            clean_str = math_to_py_syntax(eq_str)
            lhs_str, rhs_str = clean_str.split("==")
            lhs = sym.parse_expr(lhs_str, local_dict=local_dict)
            rhs = sym.parse_expr(rhs_str, local_dict=local_dict)
            
            if lhs in sys_vars:
                branch['mapping'][lhs] = rhs
            else:
                branch['basis'].append(lhs - rhs)

        # Inequalities (!=)
        for neq_str in item['inequalities']:
            clean_str = math_to_py_syntax(neq_str)
            lhs_str, rhs_str = clean_str.split("!=")
            lhs = sym.parse_expr(lhs_str, local_dict=local_dict)
            rhs = sym.parse_expr(rhs_str, local_dict=local_dict)
            branch['nonnull'].append(lhs - rhs)

        clean_branches.append(branch)

    return clean_branches


def compare_results(original_eqs, user_branches, math_branches, verbose=True):
    """
    Verifies equivalence by checking:
    1. VALIDITY: Every User branch fits inside a Math family.
    2. COVERAGE: Every Math branch is contained within a User branch.
    """
    
    def is_subset(child, parent, strict_constraints=True):
        """
        Checks if 'child' is a mathematical subset of 'parent'.
        Returns True if Parent contains Child.
        """
        # 1. BUILD & SANITIZE CHILD DEFINITION
        child_map = child['mapping'].copy() if child['mapping'] else {}
        raw_basis = child.get('basis', [])
        
        # Sanitize Child Basis (Clear denominators)
        child_basis = []
        for b in raw_basis:
            num, den = b.as_numer_denom()
            child_basis.append(num)

        # 2. DEFINE PARENT EQUATIONS
        parent_eqs = parent.get('basis', []).copy()
        if parent['mapping']:
            for lhs, rhs in parent['mapping'].items():
                parent_eqs.append(lhs - rhs)

        # 3. VERIFY PARENT EQUATIONS
        for eq in parent_eqs:
            # A. Deep Substitute Child's Mapping
            val = eq
            for _ in range(len(child_map) + 2):
                new_val = val.subs(child_map)
                if new_val == val: break
                val = new_val
            
            val = val.simplify()
            
            if val == 0:
                continue

            # B. Prepare for Ideal Membership Check
            numer, denom = val.as_numer_denom()
            
            # --- NEW: GENERATOR DISCOVERY ---
            # 1. Start with atomic symbols (Z10, Z20...)
            syms_in_expr = numer.free_symbols.union(denom.free_symbols)
            syms_in_basis = set().union(*[b.free_symbols for b in child_basis])
            base_gens = syms_in_expr.union(syms_in_basis)

            # 2. Scan for Non-Polynomial Terms (Radicals / Fractional Powers)
            #    We must treat sqrt(x) as a unique generator 'G'
            radicals = set()
            
            # Helper to scan a single expression for radicals
            def collect_radicals(expr):
                # atoms(Pow) catches x**0.5, x**-2, etc.
                for term in expr.atoms(sym.Pow):
                    # If exponent is not an integer (e.g. 0.5, 1/2), it's a radical
                    if not term.exp.is_integer:
                        radicals.add(term)
            
            collect_radicals(numer)
            collect_radicals(denom)
            for b in child_basis:
                collect_radicals(b)
                
            # 3. Combine All Generators
            all_gens = sorted(list(base_gens.union(radicals)), key=str)
            
            # --------------------------------

            # If constants remain (no variables), simple zero check
            if not all_gens:
                if val != 0: return False
                continue

            # C. Denominator Safety Check
            if denom != 1:
                # reduced returns (quotients_list, remainder)
                qs_d, r_d = sym.reduced(denom, child_basis, *all_gens)
                
                if r_d == 0:
                    # The Child forces the denominator to zero (Singularity)
                    return False

            # D. Numerator Ideal Check
            qs, r = sym.reduced(numer, child_basis, *all_gens)
            
            if r != 0:
                return False

        # 4. CHECK NONNULL CONSTRAINTS
        if strict_constraints:
            for constr in parent.get('nonnull', []):
                if constr == 1 or constr == True: continue
                
                val = constr.subs(child_map).simplify()
                
                if val == 0:
                    return False
                
        return True

    # --- Helper: Check if Branch satisfies Original Eqs ---
    def is_valid_solution(branch, equations):
        # Build full substitution map
        sol_map = branch['mapping'].copy() if branch['mapping'] else {}
        # Add basis elements = 0
        for b in branch.get('basis', []):
            if isinstance(b, sym.Symbol): sol_map[b] = 0
            
        # Verify every original equation
        for eq in equations:
            # We must simplify to handle complex algebra cancellations
            val = sym.simplify(eq)
            while any(val.has(var) for var in sol_map.keys()):
                # 1. Perform substitution
                for var, sub_val in sol_map.items():
                    val = sym.simplify(val.subs(var, sub_val))
                print(f"    [DEBUG] Substituted eq to {val} with map {sol_map}")
            if val != 0:
                print(f"    [DEBUG] Eq {eq} evaluates to {val} under branch {branch['id']} with map {sol_map}")
                return False
        return True

    print(f"\n{'='*80}")
    print(f"CENSUS & GROUND TRUTH CHECK")
    print(f"{'='*80}\n")

    # --- STEP 1: Census (User vs Math) ---
    print("--- STEP 1: Verifying against Mathematica ---")
    valid_user_count = 0
    new_solution_count = 0
    
    for u_br in user_branches:
        match_found = False
        match_type = ""
        parent_id = None
        
        # Check against Math
        for m_br in math_branches:
            if is_subset(u_br, m_br, strict_constraints=True):
                match_found = True
                parent_id = m_br['id']
                if is_subset(m_br, u_br, strict_constraints=True):
                    match_type = "[EXACT MATCH]"
                else:
                    match_type = "[SUBSET]"
                break
        
        if match_found:
            valid_user_count += 1
            print(f"  [OK] User {u_br['id']} matches Math {parent_id} {match_type}")
        else:
            # ORPHAN DETECTED: RUN GROUND TRUTH CHECK
            if is_valid_solution(u_br, original_eqs):
                new_solution_count += 1
                valid_user_count += 1
                print(f"  [WIN] User {u_br['id']} is a VALID NEW SOLUTION (Mathematica missed it)")
                print(f"        nonnull: {u_br.get('nonnull', [])}")
                print(f"        mapping: {u_br.get('mapping', {})}")
            else:
                print(f"  [FAIL] User {u_br['id']} is INVALID (Matches neither Math nor Eqs)")
                print(f"        nonnull: {u_br.get('nonnull', [])}")
                print(f"        mapping: {u_br.get('mapping', {})}")

    # --- STEP 2: Coverage (Math vs User) ---
    print(f"\n--- STEP 2: Checking Coverage of Math Families ---")
    covered_math_count = 0
    
    for m_br in math_branches:
        status = "FAIL"
        covering_users = []
        match_kind = ""
        
        for u_br in user_branches:
            if is_subset(u_br, m_br, strict_constraints=True):
                if is_subset(m_br, u_br, strict_constraints=True):
                    status = "EXACT"
                elif status != "EXACT":
                    status = "COVERED"
                covering_users.append(u_br['id'])
            elif is_subset(m_br, u_br, strict_constraints=False):
                if status == "FAIL": status = "IMPLICIT"
                covering_users.append(u_br['id'])
        
        if status != "FAIL":
            covered_math_count += 1
            print(f"  [OK] Math {m_br['id']} ({status}) covered by {list(set(covering_users))}")
        else:
            print(f"  [FAIL] Math {m_br['id']} NOT covered")

    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"Matched Mathematica: {valid_user_count - new_solution_count}")
    if new_solution_count > 0:
        print(f"    Valid Solutions not in Mathematica: {new_solution_count}")
    print(f"Mathematica Coverage: {covered_math_count}/{len(math_branches)}")
    
    successs = valid_user_count == len(user_branches) and covered_math_count == len(math_branches)
    if successs:
        print("\n[SUCCESS] SOLVER IS MATHEMATICALLY COMPLETE.")
    else:
        print("\n[WARNING] DISCREPANCIES FOUND.")

    return successs


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
        branches = parse_singular_output(raw_output, all_var_names=["x","y", "a", "b"],
                                         inv_dummy_map={'x':'x', 'y':'y', 'a':'a', 'b':'b'})
        assert len(branches) == 3 # One branch has zero solutions
        assert branches[0]['vars'] == [sym.Symbol('x'), sym.Symbol('y')]
        assert branches[0]['num_solutions'] == 1
        assert branches[2]['constraints'] == [sym.Symbol('b'), sym.Symbol('a')]


    def test_parse_no_star(self):
        raw_output = """|||START||||||BRANCH|||1
        |||COMPONENT|||1
        |||PARAMS|||a,c
        |||VARS|||b
        |||CONSTRAINTS|||
        |||NUM_CONSTRAINT_SOLUTIONS|||-1
        |||NONNULL|||
        (a+1)
        |||BASIS|||
        (a+1)*b+(ac)
        |||NUM_SOLUTIONS|||1
        |||BRANCH|||2
        |||COMPONENT|||1
        |||PARAMS|||a,c
        |||VARS|||b
        |||CONSTRAINTS|||
        (a+1)
        |||NUM_CONSTRAINT_SOLUTIONS|||1
        |||CONSTRAINT_BASIS|||
        a+1
        |||NONNULL|||
        (c),(a+1)
        |||BASIS|||
        1
        |||NUM_SOLUTIONS|||0
        |||BRANCH|||3
        |||PARAMS|||a,c
        |||VARS|||b
        |||CONSTRAINTS|||
        (c)
        (a+1)
        |||NUM_CONSTRAINT_SOLUTIONS|||1
        |||CONSTRAINT_BASIS|||
        c
        a+1
        |||NONNULL|||
        1
        |||BASIS|||
        |||NUM_SOLUTIONS|||-1
        |||END|||"""
        branches = parse_singular_output(raw_output,all_var_names=['a','b','c'],
                                         inv_dummy_map={'a':'a', 'b':'b', 'c':'c'})
        assert len(branches) == 3 # One branch has zero solutions
        assert sym.simplify(branches[0]['basis'][0] - ((symbols('a') + 1)*symbols('b') + symbols('a')*symbols('c'))) == 0


class TestSingularConsistency:
    
    def test_trivial_consistent(self):
        """Simple point solution (finite)."""
        x, y = sym.symbols('x y')
        eqs = [x - 1, y - 2]
        assert is_compatible(eqs) is True

    def test_trivial_inconsistent(self):
        """Direct mathematical contradiction."""
        assert is_compatible([sym.Integer(1)]) is False

    def test_algebraic_inconsistency(self):
        """Hidden contradiction via Groebner Basis."""
        x, y = sym.symbols('x y')
        # Parallel planes: x+y=1 and x+y=2
        eqs = [x + y - 1, x + y - 2]
        assert is_compatible(eqs) is False

    def test_infinite_solutions(self):
        """
        CRITICAL: Infinite solutions (vdim = -1).
        Must return True (Consistent).
        """
        x, y = sym.symbols('x y')
        eqs = [x * y] 
        assert is_compatible(eqs) is True

    def test_complex_solutions(self):
        """System solvable only over Complex numbers."""
        x = sym.symbols('x')
        eqs = [x**2 + 1]
        assert is_compatible(eqs) is True

    def test_overdetermined_consistent(self):
        """More equations than variables, but valid."""
        x = sym.symbols('x')
        eqs = [x**2 - 1, x - 1] # x=1 is valid
        assert is_compatible(eqs) is True

    def test_many_variables(self):
        """Test near the mapping limit."""
        # 20 variables
        vars = sym.symbols(' '.join([f'Z{i}' for i in range(20)]))
        eqs = [sum(vars)]
        assert is_compatible(eqs) is True

    def test_empty_system(self):
        """No equations is technically satisfied by everything."""
        assert is_compatible([]) is True

    def test_variable_overflow(self):
        """Ensure error raised if system exceeds Super Ring size."""
        # 50 vars is > our defined ring
        vars = sym.symbols(' '.join([f'V{i}' for i in range(60)]))
        eqs = [v - 1 for v in vars]
        
        with pytest.raises(ValueError):
            is_compatible(eqs)

    def test_difficult_systems(self):
        
        Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02 = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02')
        eq_set = [Z00*Z11 - 2*Z10*Z11 + Z10*Z21 + Z11*Z20 - 2*Z20*Z21,
                    Z00*Z22*(Z12 - Z22) + 2*Z12**2*Z20 - 2*Z12*Z20*Z22 + 2*Z20*Z22**2,
                    Z10*Z22*(Z12 - Z22) + Z12**2*Z20 - Z12*Z20*Z22 + 2*Z20*Z22**2,
                    Z11*(2*Z12 - Z22) - Z12*Z21 + 2*Z21*Z22]
        assert is_compatible(eq_set) is True

        Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02 = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02')
        eq_set = [Z00*Z12 - Z10*Z12 - Z20*Z22,
                    Z11*Z12 + Z21*Z22, 
                    Z00*Z11 - 2*Z10*Z11 + Z10*Z21 + Z11*Z20 - 2*Z20*Z21]
        assert is_compatible(eq_set) is True

    def test_fraction_inconsistency(self):
        """System with fractions that is inconsistent."""
        x, y = sym.symbols('x y')
        eqs = [1/x + 1/y, x + y - 1]
        assert is_compatible(eqs) is False

    def test_fraction_consistency(self):
        """System with fractions that is consistent."""
        x, y = sym.symbols('x y')
        eqs = [1/x + 1/y - 1, x*y - 2]
        assert is_compatible(eqs) is True


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
            mappings = b['mapping']
            constraints = b['constraints']
            
            # Check for Generic Case
            if self.x in mappings and mappings[self.x] == self.b / self.a:
                has_generic = True
            
            # Check for Singular Case (Constraints contain a and b)
            # Note: exact constraint check can be tricky due to formatting, 
            # but usually it's [b, a] or similar.
            if constraints and self.a in mappings and self.b in mappings and self.x in b["free_vars"]:
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
        
        solutions = [b['mapping'][self.x] for b in branches]
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
            m = b['mapping']
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
        branches = solve_with_singular(eqs, [self.x, self.y, self.z], rational_only=False)
        assert len(branches) == 6, "Cyclic-3 should have 6 solutions"
        unique_base = set()
        for b in branches:
            base = b["id"].split(".")[0]
            unique_base.add(base)
        assert len(unique_base) == 3
        
        # Check if the mappings are discrete (no free parameters)
        # Note: Cyclic-3 is 0-dimensional, so all vars should be mapped to numbers/algebraic values.
        for b in branches:
            assert b["free_vars"] == [], "Cyclic-3 solutions should have no free variables"
            assert b["vars"] == [self.x, self.y, self.z], "Cyclic-3 should have x,y,z as variables"


    def test_06_algebraic_number(self):
        """
        Tests if Singular returns the reduced algebraic form properly.
        """
        self.setup_method()
        print("\n--- Test 06: Algebraic Number (x^2 - 2 = 0) ---")
        eqs = [self.x**2 - 2]
        branches = solve_with_singular(eqs, [self.x])
        assert len(branches) == 2
        for b in branches:
            for var, val in b['mapping'].items():
                assert var == self.x
                assert val == sym.sqrt(2) or val == -sym.sqrt(2), "Algebraic number solution incorrect"


    def test_07_vs_mathematica_1(self):

        """
        
        """
        Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02 = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02')
        sys_vars =  Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02
        math_branches = parse_mathematica_reduce_json(str(Path(__file__).with_name("test01.json")), sys_vars=sys_vars)
        
        eq_set = [Z00*Z11 - 2*Z10*Z11 + Z10*Z21 + Z11*Z20 - 2*Z20*Z21,
                    Z00*Z22*(Z12 - Z22) + 2*Z12**2*Z20 - 2*Z12*Z20*Z22 + 2*Z20*Z22**2,
                    Z10*Z22*(Z12 - Z22) + Z12**2*Z20 - Z12*Z20*Z22 + 2*Z20*Z22**2,
                    Z11*(2*Z12 - Z22) - Z12*Z21 + 2*Z21*Z22]
        branches = solve_with_singular(eq_set)
        success = compare_results(eq_set, branches, math_branches, verbose=True)
        assert success, "Solver results do not match Mathematica benchmark."

    def test_08_vs_mathematica_2(self):

        """
        
        """
        Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02 = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02')
        sys_vars =  Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02
        math_branches = parse_mathematica_reduce_json(str(Path(__file__).with_name("test02.json")), sys_vars=sys_vars)
        
        eq_set = [Z00*Z12 - Z10*Z12 - Z20*Z22,
                    Z11*Z12 + Z21*Z22, 
                    Z00*Z11 - 2*Z10*Z11 + Z10*Z21 + Z11*Z20 - 2*Z20*Z21]
        branches = solve_with_singular(eq_set)
        success = compare_results(eq_set, branches, math_branches, verbose=True)

        assert success, "Solver results do not match Mathematica benchmark."

    def test_09_duplicate_eqs(self):
        
        Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02 = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02')
        eqs=(Z00 - Z10, Z00 - Z10)
        solve_vars=[Z00, Z10, Z11]
        branches = solve_with_singular(eqs, solve_vars)
        assert len(branches) == 1
        branches = solve_with_singular(eqs)
        assert len(branches) == 1

    def test_10_vs_mathematica_3(self):

        Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02, nzVar = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02, nzVar')
        sys_vars =  Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02, nzVar
        
        eqs = [Z00*Z11*Z22**2 - Z00*Z12*Z21*Z22 - Z10*Z11*Z22**2 + Z10*Z12*Z21*Z22 + Z11*Z12*Z20*Z22 - Z12**2*Z20*Z21,
               -Z00*Z11*Z21*Z22 + Z00*Z12*Z21**2 + Z10*Z11*Z21*Z22 - Z10*Z12*Z21**2 - Z11**2*Z20*Z22 + Z11*Z12*Z20*Z21,
               -Z00**2*Z21*Z22 + 2*Z00*Z10*Z21*Z22 - Z00*Z11*Z20*Z22 - Z00*Z12*Z20*Z21 - Z10**2*Z21*Z22 + Z10*Z11*Z20*Z22 + Z10*Z12*Z20*Z21 - Z11*Z12*Z20**2,
               -Z00**2*Z11*Z12,
               -Z00**2*Z21*Z22,
               -Z00*nzVar*(Z11*Z22 - Z12*Z21) + 1]
        math_branches = parse_mathematica_reduce_json(str(Path(__file__).with_name("test03.json")), sys_vars=sys_vars)
        branches = solve_with_singular(eqs)
        success = compare_results(eqs, branches, math_branches, verbose=True)
        assert success, "Solver results do not match Mathematica benchmark."

    def test_11_simple_lowercase(self):
        x, y, nzVar = sym.symbols('x y nzVar')
        eqs = [x - y - 1, -nzVar*(x - y) + 1]
        branches = solve_with_singular(eqs)
        assert len(branches) == 1

    def test_12_with_fractions(self):
        self.setup_method()
        eqs = [self.a - self.b/self.x]
        branches = solve_with_singular(eqs)
        
        # Expectation: 
        # Only the a = b case
        assert len(branches) == 2

    def test_13_difficult_rational(self):
        
        Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02, nzVar = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02 nzVar')
        vars = [Z01, Z02, Z12, nzVar]
        params = [Z11, Z21, Z22]
        eqs = [Z11*Z12 + Z21*Z22,
               -5*Z01*Z12**2 - 2*Z01*Z12*Z22 - 2*Z01*Z22**2 + 5*Z02*Z11*Z12 + Z02*Z11*Z22 + Z02*Z12*Z21 + 2*Z02*Z21*Z22 - 2*Z11*Z12*Z22 - Z11*Z22**2 + 2*Z12**2*Z21 + Z12*Z21*Z22,
               -10*Z01*Z12**2 - 2*Z01*Z12*Z22 - 9*Z01*Z22**2 + 10*Z02*Z11*Z12 + Z02*Z11*Z22 + Z02*Z12*Z21 + 9*Z02*Z21*Z22 - 5*Z11*Z12*Z22 - 4*Z11*Z22**2 + 5*Z12**2*Z21 + 4*Z12*Z21*Z22,
               -4*Z01*Z12**2 - 8*Z01*Z22**2 + 4*Z02*Z11*Z12 + 8*Z02*Z21*Z22 - 2*Z11*Z12*Z22 - 4*Z11*Z22**2 + 2*Z12**2*Z21 + 4*Z12*Z21*Z22,
               -nzVar*(Z11*Z22 - Z12*Z21) + 1]
        sols = solve_with_singular(eqs, rational_only=True)
        assert len(sols) == 0

def test_simple_subset_removal():
    """
    Test that a specific solution (all vars=0) is removed if it 
    is a subset of a general solution (Z20 free, others=0).
    """
    Z00, Z10, Z20, Z11, Z12, Z21, Z22 = symbols('Z00 Z10 Z20 Z11 Z12 Z21 Z22')
    # 1. The General Branch
    b_gen = {
        'id': 1,
        'basis': [Z11, Z12, Z21, Z22], 
        'mapping': {Z00: 0, Z10: 0}, 
        'nonnull': [1], 
        'params': [Z20] 
    }

    # 2. The Specific Branch
    b_spec = {
        'id': 2,
        'basis': [Z11, Z12, Z21, Z22, Z20, Z00, Z10],
        'mapping': {
            Z00: 0, Z10: 0, Z20: 0,   
            Z11: 0, Z12: 0, Z21: 0, Z22: 0
        },
        'nonnull': [1],
        'params': []
    }

    branches = [b_gen, b_spec]
    filtered = filter_redundant_branches(branches)

    ids = [b['id'] for b in filtered]

    assert 1 in ids
    assert 2 not in ids

def test_keep_singularity_filling_branch():
    """
    Test that a subset is KEPT if it violates the 'nonnull' constraint 
    of the general branch. 
    (i.e., The general branch has a hole, and the specific branch fills it).
    """
    # 1. General Branch: Valid ONLY if Z22 != 0
    # Equation: Z12 = 5/Z22 (Hypothetically) -> Implies Z12*Z22 - 5 = 0
    # Let's use a simpler one: Z12 = Z22, but valid only if Z22 != 0
    Z00, Z10, Z20, Z11, Z12, Z21, Z22 = symbols('Z00 Z10 Z20 Z11 Z12 Z21 Z22')
    b_gen = {
        'id': 1,
        'basis': [Z12 - Z22],
        'mapping': [],
        'nonnull': [Z22], # CONSTRAINT: Z22 cannot be 0
        'params': [Z22]
    }

    # 2. Specific Branch: The Singularity (Z12=0, Z22=0)
    # This satisfies the equation (0 - 0 = 0), BUT...
    # It sets the constraint variable (Z22) to 0.
    b_spec = {
        'id': 2,
        'basis': [Z12, Z22],
        'mapping': {'Z12': 0, 'Z22': 0},
        'nonnull': [1],
        'params': []
    }

    branches = [b_gen, b_spec]
    filtered = filter_redundant_branches(branches)

    # Assertions
    ids = [b['id'] for b in filtered]
    assert 1 in ids
    assert 2 in ids, "The specific branch should be KEPT because it fills the Z22=0 hole."
    assert len(filtered) == 2

def test_branch_26_consumes_branch_27():
    """
    Regression Test:
    Branch 26 (Decoupled Mode) allows Z00, Z10, Z20 to be free parameters, 
    provided all interaction terms (Z11, Z12, Z21, Z22) are zero.
    
    Branch 27 (Trivial Zero) has ALL variables (including energies) set to zero.
    
    B27 should be detected as a strict subset of B26.
    """
    Z00, Z10, Z20, Z11, Z12, Z21, Z22 = symbols('Z00 Z10 Z20 Z11 Z12 Z21 Z22')
    # Branch 26: The "Master" Decoupled Solution
    # Logic: Interactions are 0. Energies (Z00, Z10, Z20) are NOT in mapping/basis, so they are free.
    b26 = {
        'id': 26,
        'component': 2,
        # Implicitly: Z00, Z10, Z20 are free parameters
        'params': [Z00, Z10, Z20], 
        # Basis implies interactions are 0
        'basis': [Z22, Z21, Z12, Z11], 
        'mapping': {Z11: 0, Z12: 0, Z21: 0, Z22: 0}, 
        'nonnull': [1]
    }

    # Branch 27: The "Trivial" All-Zero Solution
    # Logic: Everything is 0.
    b27 = {
        'id': 27,
        'component': 3,
        'params': [], 
        # Basis implies everything is 0
        'basis': [Z00, Z10, Z20, Z11, Z12, Z21, Z22], 
        'mapping': {
            Z00: 0, Z10: 0, Z20: 0,
            Z11: 0, Z12: 0, Z21: 0, Z22: 0
        },
        'nonnull': [1]
    }

    branches = [b26, b27]
    filtered = filter_redundant_branches(branches)

    ids = [b['id'] for b in filtered]
    
    # B26 (General) should survive
    assert 26 in ids
    # B27 (Specific) should be removed
    assert 27 not in ids


def test_extract_mappings():
    """
    Test that mappings are extracted correctly and the real_only filter works.
    """
    # Create dummy branches with SymPy objects
    b1 = {
        'id': 1,
        'mapping':
            {sym.Symbol('x'): sym.sympify(1), sym.Symbol('y'): sym.sympify(2)},          # Real
    }
    b2 = {
        'id': 2,
        'mapping': 
            {sym.Symbol('x'): sym.sympify(5), sym.Symbol('y'): sym.I*sym.sympify(5)}           # Imaginary
    }
    
    branches = [b1, b2]

    # Case A: Extract All
    all_maps = extract_mappings(branches, real_only=False)
    assert len(all_maps) == 2

    # Case B: Real Only
    real_maps = extract_mappings(branches, real_only=True)
    assert len(real_maps) == 1
  

def test_eq_as_numer_denom():
    x, y = sym.symbols("x y")
    numer, denom = _eq_as_numer_denom(sym.Eq(1/(x + 1), 0))
    assert numer == 1
    assert denom == x + 1

    numer, denom = _eq_as_numer_denom(sym.Eq(x/y, 2))
    assert sym.simplify(numer - (x - 2*y)) == 0
    assert denom == y

    numer, denom = _eq_as_numer_denom(sym.Eq((x + 1)/(x - 1), (y + 1)/(y - 1)))
    assert denom == (x - 1)*(y - 1)


class TestSageGroebnerBridge:
    def test_returns_sympy_objects(self):
        """Ensure the function converts Sage types back to SymPy types."""
        x, y = sym.symbols('x y')
        eqs = [x**2 - 1, y - x]
        
        basis = get_sage_groebner_basis(eqs, [x, y])
        
        assert isinstance(basis, list)
        assert len(basis) > 0
        # Check that the result elements are actual SymPy objects, not Sage objects
        assert isinstance(basis[0], sym.Basic) 

    def test_groebner_correctness(self):
        """Test a known Groebner basis result."""
        x, y, z = sym.symbols('x y z')
        # Simple system: x=1, y=2, z=3
        eqs = [x - 1, y - 2, z - 3]
        
        basis = get_sage_groebner_basis(eqs, [x, y, z])
        
        # For this simple system, the basis should just be the equations themselves 
        # (normalized to monic, which they already are).
        # We compare string representations or subtract to check equivalence
        assert len(basis) == 3
        # Check if x-1 is in the basis
        assert any(sym.simplify(b - (z - 3)) == 0 for b in basis)


class Test0DSolver:
    def test_branching_rationals(self):
        """
        Test that it finds multiple rational roots.
        Equation: x^2 - 4 = 0  =>  x = 2, x = -2
        """
        x = symbols('x')
        eqs = [sym.Eq(x**2, 4)]
        
        results = solve_0D_backsub(eqs, [x])
        
        assert len(results) == 2
        # Check that both 2 and -2 are in the results
        values = [r[x] for r in results]
        assert 2 in values
        assert -2 in values

    def test_irrational_filtering(self):
        """
        Test that it strictly ignores irrational roots.
        Equation: x^2 - 2 = 0  =>  x = sqrt(2) (Irrational)
        """
        x = symbols('x')
        eqs = [sym.Eq(x**2, 2)]
        
        results = solve_0D_backsub(eqs, [x], rational_only=True)
        
        # Should be empty
        assert results == []

        results = solve_0D_backsub(eqs, [x], rational_only=False)
        
        # Should be +/- sqrt(2)
        for res in results:
            for val in res.values():
                assert abs(val**2 - 2) < 1e-09

    def test_mixed_factors(self):
        """
        Test a polynomial with both rational and irrational factors.
        Equation: (x^2 - 1)(x^2 - 2) = 0
        Roots: 1, -1 (Rational) AND sqrt(2), -sqrt(2) (Irrational)
        Result: Should return only 1 and -1.
        """
        x = symbols('x')
        eqs = [sym.Eq((x**2 - 1) * (x**2 - 2), 0)]
        
        results = solve_0D_backsub(eqs, [x], rational_only=True)
        assert len(results) == 2
        values = [r[x] for r in results]
        assert 1 in values
        assert -1 in values
        # Ensure no sqrt(2) leaked in
        assert all(v.is_rational for v in values)

        results = solve_0D_backsub(eqs, [x], rational_only=False)
        
        assert len(results) == 4
        values = [r[x] for r in results]
        assert 1 in values
        assert -1 in values
        # Ensure sqrt(2) present
        for val in values:
            if abs(val) != 1:
                assert abs(val**2 - 2) < 1e-09

    def test_system_branching(self):
        """
        Test branching in a multi-variable system.
        x^2 = 1      => x = 1, -1
        y = x + 1
        Solutions: (1, 2) and (-1, 0)
        """
        x, y = symbols('x y')
        eqs = [
            sym.Eq(x**2, 1),
            sym.Eq(y, x + 1)
        ]
        
        results = solve_0D_backsub(eqs, [x, y])
        assert len(results) == 2
        
        # Verify Solution 1: (1, 2)
        assert any(r[x] == 1 and r[y] == 2 for r in results)
        
        # Verify Solution 2: (-1, 0)
        assert any(r[x] == -1 and r[y] == 0 for r in results)

    def test_parameters_simple(self):
        """
        Test solving with parameters (Rational Functions).
        a*x - 6 = 0  =>  x = 6/a
        """
        x, a = symbols('x a')
        eqs = [sym.Eq(a * x, 6)]
        
        results = solve_0D_backsub(eqs, [x], sympy_params=[a])
        
        assert len(results) == 1
        val = results[0][x]
        assert sym.simplify(val - 6/a) == 0

    def test_parameters_branching(self):
        """
        Test that parameters branch correctly when factorable.
        x^2 - a^2 = 0  =>  (x - a)(x + a) = 0
        Solutions: x = a, x = -a
        """
        x, a = symbols('x a')
        eqs = [sym.Eq(x**2, a**2)]
        
        results = solve_0D_backsub(eqs, [x], sympy_params=[a])
        
        assert len(results) == 2
        values = [r[x] for r in results]
        
        # Use simplify to check symbolic equality
        assert any(sym.simplify(v - a) == 0 for v in values)
        assert any(sym.simplify(v + a) == 0 for v in values)

    def test_parameters_irrational(self):
        """
        Test that radicals in parameters are included
        x^2 - a = 0  =>  x = sqrt(a)
        """
        x, a = symbols('x a')
        eqs = [sym.Eq(x**2, a)]
        
        results = solve_0D_backsub(eqs, [x], sympy_params=[a], rational_only=True)
        assert len(results) == 2
        values = [r[x] for r in results]
        assert any(sym.simplify(v - sym.sqrt(a)) == 0 for v in values)
        assert any(sym.simplify(v + sym.sqrt(a)) == 0 for v in values)


    def test_no_solution_inconsistent(self):
        """Test an inconsistent system returns empty list."""
        x = symbols('x')
        eqs = [sym.Eq(x, 1), sym.Eq(x, 2)]
        
        results = solve_0D_backsub(eqs, [x])
        assert results == []

    def test_high_degree_rational(self):
        """
        Test that high degree polynomials work if they have integer roots.
        x^3 - 1 = 0 => (x-1)(x^2+x+1) = 0
        Rational Root: x = 1
        Complex Roots: (-1 +/- i*sqrt(3))/2 (Should be ignored)
        """
        x = symbols('x')
        eqs = [sym.Eq(x**3 - 1, 0)]
        
        results = solve_0D_backsub(eqs, [x], rational_only=True)
        
        assert len(results) == 1
        assert results[0][x] == 1

    def test_infinite_solutions_error(self):
        """Test that infinite solutions raise the specific error."""
        x, y = var('x, y')
        # x + y = 0 has infinite solutions
        eqs = [x + y == 0]
        
        with pytest.raises(ValueError, match="Unconstrained variable"):
            solve_0D_backsub(eqs, [x, y])

    def test_difficult_rational(self):
        
        Z10, Z01, Z11, Z12, Z22, Z21, Z00, Z20, Z02, nzVar = sym.symbols('Z10 Z01 Z11 Z12 Z22 Z21 Z00 Z20 Z02 nzVar')
        vars = [Z01, Z02, Z12, nzVar]
        params = [Z11, Z21, Z22]
        eqs = [Z22**3*nzVar**3*(Z11**3 - 15*Z11**2*Z21 + 3*Z11*Z21**2 + 2*Z21**3) + 3*Z22**2*nzVar**2*(-Z11**2 + 10*Z11*Z21 - Z21**2) + 3*Z22*nzVar*(Z11 - 5*Z21) - 1,
               2*Z11*Z22 + Z12*Z21 - 15*Z21*Z22 + Z22**3*nzVar**2*(Z11**3 - 15*Z11**2*Z21 + 3*Z11*Z21**2 + 2*Z21**3) - 3*Z22**2*nzVar*(Z11**2 - 10*Z11*Z21 + Z21**2),
               6*Z02*Z21**2 + 2*Z11**2*Z22 - 30*Z11*Z21*Z22 + Z12*Z21*(2*Z11 - 11*Z21) - 2*Z21**2*Z22 - 2*Z22**2*nzVar*(Z11**3 - 15*Z11**2*Z21 + 3*Z11*Z21**2 + 2*Z21**3),
               6*Z01*Z21 + Z11**2 - 11*Z11*Z21 - 5*Z21**2 - Z22*nzVar*(Z11**3 - 15*Z11**2*Z21 + 3*Z11*Z21**2 + 2*Z21**3)]
        sols = solve_0D_backsub(eqs, vars, params, rational_only=True)
        assert len(sols) == 0

class TestSingularLifecycle:
    
    @pytest.fixture
    def solver(self):
        """Fixture to create and clean up the solver."""
        # Use a minimal startup code for speed
        s = SafeSingular(startup_code="ring r=0,(x,y),dp;")
        yield s
        # Final cleanup after test finishes
        s.kill()

    def test_timeout_kills_process_completely(self, solver):
        """
        Verifies that a timeout triggers a hard kill of the underlying PID.
        """
        # 1. Start the solver and get the initial PID
        solver.start()
        assert solver.process is not None
        old_pid = solver.process.pid
        
        print(f"\n[Test] Initial Singular PID: {old_pid}")
        
        # Verify the process actually exists in the OS
        assert psutil.pid_exists(old_pid)
        
        # 2. Run a command guaranteed to hang (Infinite Loop)
        # "while(1){ 1; }" forces Singular to spin forever without output
        print("[Test] Sending infinite loop command...")
        
        with pytest.raises(TimeoutError):
            # Set a short timeout (e.g., 1 second)
            solver.eval("while(1){ 1; }", timeout=1.0)
            
        print("[Test] Timeout caught successfully.")

        # 3. VERIFY DEATH: The old PID should no longer exist
        # We give the OS a tiny moment to update the process table, though wait() in kill() should be synchronous.
        time.sleep(0.1) 
        
        is_alive = psutil.pid_exists(old_pid)
        
        # If psutil says it exists, check if it's just a "zombie" (dead but not reaped)
        # Note: Your class calls .wait(), so it should be fully gone.
        if is_alive:
            try:
                p = psutil.Process(old_pid)
                status = p.status()
                print(f"[Test] Process status: {status}")
                # If it's running/sleeping, that's a FAIL. If it's zombie/dead, it's acceptable (but ideally gone).
                assert status == psutil.STATUS_ZOMBIE, f"Process {old_pid} is still {status}!"
            except psutil.NoSuchProcess:
                # It died between the check and now, which is a pass
                pass
        else:
            print(f"[Test] Process {old_pid} is confirmed dead.")

    def test_auto_healing_after_crash(self, solver):
        """
        Verifies that the solver automatically starts a NEW process 
        after the previous one was killed.
        """
        solver.start()
        pid_1 = solver.process.pid
        
        # 1. Crash it intentionally with a timeout
        with pytest.raises(TimeoutError):
            solver.eval("while(1){ 1; }", timeout=0.5)
            
        # 2. Run a valid command immediately after
        # This triggers the "healing" logic (start new process + run startup code)
        result = solver.eval("1 + 1", timeout=5)
        
        
        # 3. Assertions
        assert result.strip().split("\n")[0] == "2"
        assert solver.process is not None
        pid_2 = solver.process.pid


        
        print(f"\n[Test] PID 1: {pid_1}, PID 2: {pid_2}")
        
        assert pid_1 != pid_2, "The solver reused the same hung process ID! (It should be new)"


if __name__ == "__main__":
    # Run tests
    # test_solver = TestSingularParser()
    # test_solver.test_parse()
    # test_solver.test_parse_no_star()
    # test_solver = TestSingularSolver()
    # test_solver.test_01_parametric_singularity()
    # test_solver.test_02_reducible_geometry()
    # test_solver.test_03_inconsistent_system()
    # test_solver.test_04_mixed_dimension()
    # test_solver.test_05_cyclic_3()
    # test_solver.test_06_algebraic_number()
    # test_solver.test_07_vs_mathematica_1()
    # test_solver.test_08_vs_mathematica_2()
    # test_solver.test_09_duplicate_eqs()
    # test_solver.test_10_vs_mathematica_3()
    # test_solver.test_11_simple_lowercase()
    # test_solver.test_12_with_fractions()
    # test_solver.test_13_difficult_rational()
    # test_filter = TestRedundantBranchFilter()
    # test_filter.test_simple_subset_removal()
    # test_filter.test_keep_singularity_filling_branch()
    # test_filter.test_branch_26_consumes_branch_27()
    # test_extract_mappings()
    # test_const = TestSingularConsistency()
    # test_const.test_trivial_consistent()
    # test_const.test_trivial_inconsistent()
    # test_const.test_algebraic_inconsistency()
    # test_const.test_infinite_solutions()
    # test_const.test_complex_solutions()
    # test_const.test_overdetermined_consistent()
    # test_const.test_many_variables()
    # test_const.test_empty_system()
    # test_const.test_variable_overflow()
    # test_const.test_difficult_systems()
    # test_gb = TestSageGroebnerBridge()
    # test_gb.test_returns_sympy_objects()
    # test_gb.test_groebner_correctness()

    test_solver = Test0DSolver()
    # test_solver.test_branching_rationals()
    # test_solver.test_irrational_filtering()
    # test_solver.test_mixed_factors()
    test_solver.test_infinite_solutions_error()
    # test_solver.test_system_branching()
    # test_solver.test_parameters_simple()
    # test_solver.test_parameters_irrational()
    # test_solver.test_no_solution_inconsistent()
    # test_solver.test_high_degree_rational()
    # test_solver.test_difficult_rational()
    