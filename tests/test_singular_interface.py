"""
Tests for singular_interface module.

Tests the Groebner Cover functionality using examples from the
Singular grobcov.lib documentation.

Author: Eli Weissler
"""

import pytest
import re
import sympy as sym
from sympy import symbols, sympify


from sircuitenum.singular_interface import solve_with_singular, parse_singular_output, filter_redundant_branches
from sircuitenum.singular_interface import extract_mappings, flatten_branches, _robust_substitute


import json
import json
import sympy as sym

import json
import sympy as sym

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
    print(f"New Valid Solutions: {new_solution_count} (Complex roots likely missed by Math)")
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
        branches = solve_with_singular(eqs, [self.x, self.y, self.z])
        
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
        math_branches = parse_mathematica_reduce_json("test01.json", sys_vars=sys_vars)
        
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
        math_branches = parse_mathematica_reduce_json("test02.json", sys_vars=sys_vars)
        
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
        'mappings': [
            {sym.Symbol('x'): sym.sympify(1), sym.Symbol('y'): sym.sympify(2)},          # Real
            {sym.Symbol('x'): sym.I, sym.Symbol('y'): sym.sympify(2)}       # Complex
        ]
    }
    b2 = {
        'id': 2,
        'mappings': [
            {sym.Symbol('x'): sym.sympify(5), sym.Symbol('y'): sym.sympify(5)}           # Real
        ]
    }
    
    branches = [b1, b2]

    # Case A: Extract All
    all_maps = extract_mappings(branches, real_only=False)
    assert len(all_maps) == 3
    assert {sym.Symbol('x'): 1, sym.Symbol('y'): 2} in all_maps
    assert {sym.Symbol('x'): sym.I, sym.Symbol('y'): 2} in all_maps

    # Case B: Real Only
    real_maps = extract_mappings(branches, real_only=True)
    assert len(real_maps) == 2
    assert {sym.Symbol('x', real=True): 1, sym.Symbol('y', real=True): 2} in real_maps
    assert {sym.Symbol('x', real=True): 5, sym.Symbol('y', real=True): 5} in real_maps
    # Ensure the complex one is gone
    assert {sym.Symbol('x'): sym.I, sym.Symbol('y'): 2} not in real_maps


def test_flatten_branches():
    """
    Test that a branch with multiple mappings is exploded into 
    separate branches with unique IDs and singular 'mapping' keys.
    """
    # Branch 1 has 2 mappings (Needs flattening)
    b1 = {
        'id': 1,
        'some_data': 'A', # Ensure extra data is preserved
        'mappings': [
            {'val': 100}, 
            {'val': 200}
        ]
    }
    
    # Branch 2 has 0 mappings (Should be preserved as-is or handled gracefully)
    b2 = {
        'id': 2,
        'some_data': 'B',
        'mappings': [] 
    }

    branches = [b1, b2]
    flat = flatten_branches(branches)

    # Expect: 
    # b1 splits into "1.0" and "1.1"
    # b2 is passed through (or skipped depending on your logic, usually kept)
    
    # Check IDs
    ids = [b['id'] for b in flat]
    assert "1.0" in ids
    assert "1.1" in ids
    
    # Check Structure
    # Find the branch corresponding to the first mapping of b1
    b_flat_0 = next(b for b in flat if b['id'] == "1.0")
    
    # It should have 'mapping' (singular) matching the data
    assert b_flat_0['mapping'] == {'val': 100}
    # It should NOT have 'mappings' (plural)
    assert 'mappings' not in b_flat_0
    # It should retain other keys
    assert b_flat_0['some_data'] == 'A'


if __name__ == "__main__":
    # Run tests
    test_solver = TestSingularParser()
    test_solver.test_parse()
    test_solver.test_parse_no_star()
    test_solver = TestSingularSolver()
    # test_solver.test_01_parametric_singularity()
    # test_solver.test_02_reducible_geometry()
    # test_solver.test_03_inconsistent_system()
    # test_solver.test_04_mixed_dimension()
    # test_solver.test_05_cyclic_3()
    # test_solver.test_06_algebraic_number()
    # test_solver.test_07_vs_mathematica_1()
    # test_solver.test_08_vs_mathematica_2()
    # test_solver.test_09_duplicate_eqs()
    # test_filter = TestRedundantBranchFilter()
    # test_filter.test_simple_subset_removal()
    # test_filter.test_keep_singularity_filling_branch()
    # test_filter.test_branch_26_consumes_branch_27()
    test_extract_mappings()
    # test_flatten_branches()