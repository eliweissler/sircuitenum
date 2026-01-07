"""Equation set management and symbolic solving utilities.

This module provides the EquationSet class for managing collections of
symbolic equations with deduplication, caching, and compatibility checking.

Author: Eli Weissler
Version: 0.1.0
"""

__all__ = ["EquationSet", "maximally_compatible_set", "fully_compatible_set", "sol_indep_of_vars", "cached_solve", "extract_denom", "unique_solutions"]

from dataclasses import dataclass, field
from typing import Union, Sequence, Iterable, Mapping, Optional, Tuple
import sympy as sym
import functools, itertools
from sympy.core.mul import Mul


# Cache of solved equations
CACHE_VARS = sym.symbols(",".join([f'x{i}' for i in range(100)]), real=True)
CACHE_CONST = sym.symbols(",".join([f'c{i}' for i in range(100)]), real=True)
SOLVE_CACHE = {}
UNSOLVABLE_CACHE = set()  # sets of equations with no solutions

 
@dataclass(frozen=True, slots=True)
class EquationSet:
    """Immutable container for symbolic equation sets with deduplication.
    
    EquationSet provides a hashable, immutable representation of equation sets.
    EquationSets are order-insensitive (e.g., Eq(a,b) == Eq(b,a))
    and the set can be converted between various formats. The class will only
    have a dictionary representation if the equations are not obviously inconsistent.
    This means that they do not contain conflicting constant assignments for the same variable.
    
    Attributes
    ----------
    eqs : Tuple[sym.Eq, ...]
        Tuple of SymPy equations in the set.
    
    Examples
    --------
    >>> import sympy as sym
    >>> x, y = sym.symbols('x y')
    >>> eq_set = EquationSet.from_any([sym.Eq(x, 1), sym.Eq(y, 2)])
    >>> eq_set.as_dict()
    {x: 1, y: 2}
    """
    eqs: Tuple[sym.Eq, ...]
    solve_vars: Tuple[sym.Symbol, ...]
    _eqs_simplified: Tuple[sym.Eq, ...] = field(init=False, repr=False, default=None)
    _solve_vars_simplified: Tuple[sym.Symbol] = field(init=False, repr=False, default=None)
    _pair_key: frozenset = field(init=False, repr=False)
    _dict: Optional[dict] = field(init=False, repr=False, default=None)
    _canonical_key: frozenset = field(init=False, repr=False)
    _constant_conflict: bool = field(init=False, repr=True, default=False)

    @staticmethod
    def _coerce(raw) -> Tuple[sym.Eq, ...]:
        """Coerce various input formats to tuple of equations.
        
        Parameters
        ----------
        raw : Mapping | Iterable
            Input to coerce. Can be dict, list/tuple of equations/expressions/dicts,
            or frozenset of frozensets.
            
        Returns
        -------
        Tuple[sym.Eq, ...]
            Normalized tuple of SymPy equations.
        """
        # Handle dict → Eq
        if isinstance(raw, Mapping):
            items = [sym.Eq(k, v) for k, v in raw.items()]
        # Handle iterable of eq/expr/frozenset/tuple-of-frozenset
        elif isinstance(raw, (list, tuple, frozenset)):
            items = list(raw)
            # frozenset of frozensets
            if items and isinstance(items[0], frozenset):
                items = [sym.Eq(*tuple(fs)) for fs in items if len(fs) == 2]
            # list/tuple of expressions, set equal to 0
            elif items and isinstance(items[0], sym.Expr):
                expr_eqs = []
                for expr in items:
                    if expr.is_number:
                        if expr != 0:
                            return tuple()  # inconsistent → empty
                    else:
                        expr_eqs.append(sym.Eq(expr, 0))
                items = expr_eqs
            # list/tuple of dicts, set vals equal to keys
            elif items and isinstance(items[0], Mapping):
                dict_eqs = []
                for d in items:
                    dict_eqs.extend([sym.Eq(k, v) for k, v in d.items()])
                items = dict_eqs
            # list/tuple of equations
            elif items and isinstance(items[0], sym.Eq):
                items = items
        # Input is already an EquationSet
        elif isinstance(raw, EquationSet):
            return raw.eqs
            
        return tuple(items)

    @staticmethod
    def _pair(eq: sym.Eq):
        """Create order-insensitive frozenset key for an equation.
        
        Parameters
        ----------
        eq : sym.Eq
            SymPy equation.
            
        Returns
        -------
        frozenset
            Frozenset of (lhs, rhs) such that Eq(a,b) and Eq(b,a) produce same key.
        """
        # order-insensitive pair key so Eq(a,b) == Eq(b,a)
        lhs, rhs = eq.lhs, eq.rhs
        return frozenset((lhs, rhs))

    @classmethod
    def from_any(cls, raw, solve_vars=[]) -> "EquationSet":
        """Create EquationSet from various input formats.
        
        Parameters
        ----------
        raw : Mapping | Iterable | EquationSet
            Input equations. Accepts:
            - dict: {var: value} mappings
            - list/tuple of sym.Eq equations
            - list/tuple of expressions (converted to Eq(expr, 0))
            - frozenset of frozensets
            - existing EquationSet (returns copy)
         solve_vars: Iterable, optional
            The variables to solve for
            
        Returns
        -------
        EquationSet
            New EquationSet with normalized equations.
        """
        # if raw is None or empty, return empty EquationSet
        if raw is None:
            return cls.empty()
        if isinstance(raw, (list, tuple, frozenset)) and len(raw) == 0:
            return cls.empty()
        eqs = cls._coerce(raw)
        # normalize by removing trivially true eqs; keep ordering stable
        try:
            norm = tuple(eq for eq in eqs if not (eq.lhs.is_number and eq.rhs.is_number and eq.lhs == eq.rhs))
        except:
            breakpoint()
        if not solve_vars:
            # infer solve vars from equations
            sv = set()
            for eq in norm:
                sv.update(x for x in eq.free_symbols)
            solve_vars = sorted(sv, key=lambda x: str(x))
        obj = cls(norm, solve_vars)
       
        return obj
    
    @classmethod
    def empty(cls) -> "EquationSet":
        """Create an empty EquationSet.
        
        Returns
        -------
        EquationSet
            Empty equation set with no equations.
        """
        return cls(tuple(), tuple())

    def __post_init__(self):

        # Build dictionary representation and constant substitutions
        _dict, const_subs = self._build_dict()

        # Check to see if any constant substitutions yield undefined behavior
        # and simplify equations accordingly
        eq_simplified = []
        solve_vars_simplified = set()
        for eq in self.eqs:
            new_eq = eq.subs(const_subs)
            if isinstance(new_eq, sym.Eq):
                eq_simplified.append(new_eq)
                solve_vars_simplified.add(x for x in new_eq.free_symbols if x in self._solve_vars)
                if eq.lhs.is_finite == False or eq.rhs.is_finite == False:
                    object.__setattr__(self, "_constant_conflict", True)
            if isinstance(new_eq, sym.logic.boolalg.BooleanFalse):
                object.__setattr__(self, "_constant_conflict", True)
                continue
            elif isinstance(new_eq, sym.logic.boolalg.BooleanTrue):
                continue
        object.__setattr__(self, "_eqs_simplified", eq_simplified)
        object.__setattr__(self, "_solve_vars_simplified", eq_simplified)

        object.__setattr__(self, "_pair_key", frozenset(self._pair(eq) for eq in self.eqs))
        object.__setattr__(self, "_dict", _dict)
        # Compute canonical representation (currently no-op, returns _pair_key)
        canonical = self._compute_canonical_key()
        object.__setattr__(self, "_canonical_key", canonical)
                

    def _build_dict(self):
        """Build dictionary representation from equations.
        
        Returns
        -------
        dict or None
            Dictionary mapping variables to values, or None if equations are
            inconsistent (same variable maps to different values).
        """
        d = {}
        const_subs = {}
        dict_exists = True
        for eq in self.eqs:
            var, val = eq.lhs, eq.rhs
            # Number != Number
            if var.is_number and val.is_number:
                if var != val:
                    object.__setattr__(self, "_constant_conflict", True)
                    return None, {}
                continue
            # val is number and var is not -> swap
            if var.is_number and not val.is_number:
                var, val = val, var
            # Variable already assigned a different number
            if not var.is_number and val.is_number:
                if var in d and d[var].is_number and d[var] != val:
                    object.__setattr__(self, "_constant_conflict", True)
                    return None, {}
            if var in d and d[var] != val:
                dict_exists = False
                continue
            d[var] = val
            # Track constant substitutions for conflict checking
            if val.is_number and not var.is_number:
                const_subs[var] = val

        if dict_exists:
            return d, const_subs
        else:
            return None, const_subs
    
    def _compute_canonical_key(self) -> frozenset:
        """Compute canonical key with standardized variable names.
        
        Returns
        -------
        frozenset
            Canonical representation key.
            
        Notes
        -----
        Currently returns _pair_key unchanged. Variable canonicalization to be implemented.
        """
        """Compute canonical key with standardized variable names."""
        return frozenset((self._pair_key, frozenset(self.solve_vars)))

    def get_solve_vars(self) -> list[sym.Symbol]:
        """Get list of variables to solve for.
        
        Returns
        -------
        list[sym.Symbol]
            List of variables to solve for.
        """
        return list(self._solve_vars_simplified)

    def as_eq_list(self) -> list[sym.Eq]:
        """Convert to list of SymPy equations.
        
        Returns
        -------
        list[sym.Eq]
            List of equations in this set.
        """
        return list(self.eqs)
        # return list(self._eqs_simplified)
    
    def as_grobner_list(self) -> list[sym.Eq]:
        """Convert to list of SymPy equations in Groebner basis order.
        
        Returns
        -------
        list[sym.Eq]
            List of equations in Groebner basis order.
        """
        if len(self.eqs) == 0:
            return []
        grob_eqs = sym.groebner([eq.lhs - eq.rhs for eq in self.as_eq_list()], *self._solve_vars)
        return [sym.Eq(poly, 0) for poly in grob_eqs]

    def as_frozenset(self) -> frozenset:
        """Convert to frozenset of equation pairs.
        
        Returns
        -------
        frozenset
            Frozenset of frozensets, each containing (lhs, rhs) of an equation.
        """
        return self._pair_key

    def as_dict(self) -> Optional[dict]:
        """Convert to dictionary mapping variables to values.
        
        Returns
        -------
        dict or None
            Dictionary {var: value}, or None if equations are inconsistent.
        """
        return None if self._dict is None else dict(self._dict)

    def __hash__(self):
        return hash(self._canonical_key)

    def __eq__(self, other):
        return isinstance(other, EquationSet) and self._pair_key == other._pair_key
    
    def __len__(self):
        return len(self.eqs)
    
    def issubset(self, other: "EquationSet") -> bool:
        """Check if this EquationSet is a subset of another.
        
        Parameters
        ----------
        other : EquationSet
            EquationSet to compare against.
            
        Returns
        -------
        bool
            True if all equations in self are in other.
        """
        if not isinstance(other, EquationSet):
            return NotImplemented
        return self._pair_key.issubset(other._pair_key)
    
    def __or__(self, other: "EquationSet") -> "EquationSet":
        if not isinstance(other, EquationSet):
            return NotImplemented
        if self._pair_key == other._pair_key or len(other.eqs) == 0:
            return self
        if len(self.eqs) == 0:
            return other
        return EquationSet.from_any(self.as_eq_list() + other.as_eq_list())

    def __ior__(self, other: "EquationSet") -> "EquationSet":
        if not isinstance(other, EquationSet):
            return NotImplemented
        return self.__or__(other)
    
    def as_numer_denom(self) -> tuple["EquationSet", list]:
        """Convert equations to numerator form and extract denominators.
        
        Returns
        -------
        tuple[EquationSet, list]
            - EquationSet with equations converted to Eq(numerator, 0)
            - List of denominators extracted from each equation
        """

        numer_eqs = []
        denoms = []
        for eq in self.as_eq_list():
            numer, denom = _eq_as_numer_denom(eq)
            denoms.append(denom)
            numer_eqs.append(sym.Eq(numer, 0))
        return EquationSet.from_any(numer_eqs, solve_vars=self.solve_vars), denoms


def maximally_compatible_set(terms: Union[list[list[EquationSet]], list[list[dict]]], solve_vars: list[sym.Symbol],
                             nonzero: list[sym.Expr] = [], tiebreaker_fn=lambda x: 0):
    """
    Given a list of list of substitutions, identifies the largest set of compatible
    substitutions that include one substitution from each list.

    For example, given terms = [[{x:1}, {x:2}], [{y:3}], [{x:1, z:4}, {z:5}]],
    the function will attempt to select one substitution from each inner list such that
    all substitutions are compatible (no conflicting assignments to the same variable).
    In this case, the largest compatible set would be [{x:1}, {y:3}, {x:1, z:4}].
    
    Parameters
    ----------
    terms : list[list[EquationSet]] or list[list[dict]]
        List of term groups, each containing possible substitutions.
    solve_vars : list[sym.Symbol]
        Variables to solve for.
    nonzero : list[sym.Expr], optional
        Expressions that must be nonzero in solutions.
    tiebreaker_fn : callable, optional
        Function to rank equally-sized compatible sets. Lower values preferred.
        
    Returns
    -------
    tuple[list, list]
        - List of index tuples identifying best compatible term combinations
        - List of corresponding solution dictionaries

    """
    # Convert all terms to dictionary form
    terms = [[t.as_dict() if isinstance(t, EquationSet) else t for t in group] for group in terms]

    # Keep track of which terms are incompatible/compatible
    incompatible = set()
    compatible = {}
    best_subs = [{}]
    n_terms = len(terms)
    best_val = tiebreaker_fn([])
    best_keys = []
    best_nz = 0
    # Assume each individual term is valid
    for nz in range(1, n_terms+1):
        # Pidgeonhole principle, is it possible to to select nz keys that
        # might be compatible
        if len(incompatible) > 0:
            if nz > n_terms - max(len(incompat_keys) for incompat_keys in incompatible) + 1:
                break
        for keys in itertools.combinations(range(n_terms), nz):

            # Check if a subset of the keys are incompatible
            if any(all(k in keys for k in incompat_keys) for incompat_keys in incompatible):
                continue

            # Fill in rule set with possible simplifications
            # from previous iterations
            keys_left = [k for k in keys]
            rules_sets = []
            for keyset in sorted(compatible.keys(), key=len)[::-1]:
                if all(k in keys_left for k in keyset):
                    rules_sets.append(compatible[keyset])
                    for k in keyset:
                        keys_left.remove(k)
                if keys_left == []:
                    break
            for k in keys_left:
                rules_sets.append(terms[k])
            
            # Identify a set of rules that are fully compatible
            res = fully_compatible_set(rules_sets, solve_vars, depth_first=False)
            if len(res) == 0:
                incompatible.add(keys)
                continue
            # Verify that the nonzero entries are nonzero
            good_subs = []
            tiebreak_val = tiebreaker_fn(keys)
            for i, compat_sub in enumerate(res):
                if any(sym.simplify(d.subs(compat_sub)) == 0 for d in nonzero):
                    continue
                compat_eq = EquationSet.from_any(compat_sub, solve_vars=solve_vars)
                if compat_eq._constant_conflict:
                    continue
                if compat_eq in good_subs:
                    continue
                if (tiebreak_val < best_val and nz == best_nz) or (nz > best_nz):
                    best_val = tiebreak_val
                    best_subs = [compat_sub]
                    best_keys = [keys]
                    best_nz = nz
                elif (tiebreak_val == best_val) and (nz == best_nz):
                    best_subs.append(compat_sub)
                    best_keys.append(keys)
                good_subs.append(compat_sub)
            # Save the results for these keys
            if len(keys) > 1:
                if len(good_subs) == 0:
                    incompatible.add(keys)
                else:
                    compatible[keys] = good_subs

    return best_keys, best_subs

COMPATIBLE_CACHE = {}
def fully_compatible_set(assumptions: Union[frozenset[frozenset[EquationSet]], list[list[dict]]], solve_vars: list[sym.Symbol],
                          starting: Union[EquationSet, dict] = EquationSet.empty(), nonzero=[], depth_first=False):
    """
    Given a set of assumptions (each a set of equation sets), finds all combinations that include
    one equation set from each assumption that are mutually compatible. Rather than returning the combinations
    of equation sets, it returns the resulting substitutions as dictionaries.

    For example, given assumptions = frozenset([frozenset([eq_set1, eq_set2]), frozenset([eq_set3])]),
    the function will attempt to select one equation set from each inner frozenset that are compatible.
    If eq_set1 and eq_set3 are compatible, but eq_set2 and eq_set3 are not, the result will include only
    combinations including eq_set1.

    Returns all compatible combinations unless depth_first=True, in which case it returns the first found.
    
    Parameters
    ----------
    assumptions : frozenset[frozenset[EquationSet]] or list[list[dict]]
        Nested structure of equation sets representing assumptions.
    solve_vars : list[sym.Symbol]
        Variables to solve for.
    starting : EquationSet, optional
        Initial equation set to start from.
    nonzero : list[sym.Expr], optional
        Expressions that must be nonzero in solutions.
    depth_first : bool, optional
        If True, return first compatible set found. If False, find all.
        
    Returns
    -------
    list[dict]
        List of compatible solution dictionaries.
        
    Notes
    -----
    Results are cached for efficiency when depth_first=False.
    """


    # Convert starting to EquationSet if needed
    if not isinstance(starting, EquationSet):
        starting = EquationSet.from_any(starting, solve_vars=solve_vars)
    # Convert all assumptions to EquationSet
    if isinstance(assumptions, list):
        assumptions = frozenset(frozenset(EquationSet.from_any(t, solve_vars=solve_vars) if not isinstance(t, EquationSet) else t for t in group) for group in assumptions)

    # Base case: no more assumptions to process
    if len(assumptions) == 0:
        return [starting.as_dict()]
    
    # Starting is not a valid set of substitutions
    if starting._constant_conflict:
        return []
    
    # Check cache
    if (starting, assumptions, frozenset(nonzero)) in COMPATIBLE_CACHE:
        return COMPATIBLE_CACHE[(starting, assumptions, frozenset(nonzero))]

    # Existing equations from starting assumptions
    all_res = []
    assumptions_list = list(assumptions)
    next_assumptions = frozenset(assumptions_list[1:])
    for eq_set in assumptions_list[0]:
        # Combine the equation sets
        combined_eq_set = eq_set | starting
        # Conflict in the substitutions of constants
        if combined_eq_set._constant_conflict:
            continue
        sols = cached_solve(combined_eq_set, solve_vars)
        if len(sols) > 0:
            for sol in sols:
                # Make sure value isn't 0
                this_nz = [sym.simplify(d.subs(sol)) for d in nonzero]
                if any(d == 0 for d in this_nz):
                    continue
                res = fully_compatible_set(next_assumptions, solve_vars, starting=sol,
                                                nonzero=this_nz, depth_first=depth_first)
                if res and depth_first:
                    return res
                all_res += res
    
    # Save in cache if we're not doing depth first
    if not depth_first:
        COMPATIBLE_CACHE[(starting, assumptions, frozenset(nonzero))] = all_res

    return all_res
    

def sol_indep_of_vars(expr: Union[sym.Eq, sym.Expr], solve_vars, nonzero=[]):
    """Find solutions independent of non-solve variables.
    
    Determines whether solutions exist that depend only on specified solve
    variables, not on other free symbols in the expression.
    
    Parameters
    ----------
    expr : sym.Eq or sym.Expr
        Equation or expression to solve (expression set to 0).
    solve_vars : list[sym.Symbol]
        Variables to solve for.
    nonzero : list[sym.Expr], optional
        Expressions that must be nonzero in solutions.
        
    Returns
    -------
    list[dict]
        List of minimal solution dictionaries mapping solve_vars to values.
        
    Raises
    ------
    ValueError
        If solution does not satisfy the original expression.
    """

    # Simplify to rational form
    numer, denom = _eq_as_numer_denom(expr)

    # No solutions if there are no solve variables
    if not any(v in numer.free_symbols for v in solve_vars):
        return []

    # Get unique symbolic products of non-excluded vars
    var_combos = _unique_products(numer, exclude=solve_vars)

    # All coefficients must be able to be zero simultaneously
    coeffs = []
    # Constant term must be 0
    if 1 in var_combos:
        val = var_combos[1]
        if val.is_number and val != 0:
            return []
        else:
            coeffs.append(sym.simplify(var_combos.pop(1)))
    for v in var_combos:
        coeffs.append(sym.simplify(var_combos[v]))

    # Get solutions for each equation and
    # check compatibility with each other
    partial_sols = []
    for coeff in coeffs:
        si = cached_solve([coeff], solve_vars)
        if si:
            partial_sols.append(si)
        # At least one of them is unsolvable
        else:
            return []
    sols = unique_solutions(fully_compatible_set(partial_sols, solve_vars, nonzero=[]))

    # Make sure none of the nonzero conditions are violated
    sols = [s for s in sols if all(sym.simplify(d.subs(s)) != 0 for d in nonzero)]
    sols = [s for s in sols if sym.simplify(denom.subs(s)) != 0]

    # Verify that the original expression is indeed zero
    if any(not sym.simplify(expr.subs(s)).is_zero for s in sols):
        breakpoint()
        raise ValueError("Solution does not satisfy original expression")

    return sols

def cached_solve(all_eq: Union[EquationSet, list], solve_vars: list[sym.Symbol] = []):
    """Wrapper for _cached_solve to handle unhashable types."""
    # Convert equations to a hashable representation
    solve_vars_key = tuple(sorted(solve_vars, key=lambda x: str(x)))
    if isinstance(all_eq, EquationSet):
        eq_key = all_eq
    else:
        eq_key = EquationSet.from_any(all_eq, solve_vars=solve_vars_key)
    return _cached_solve(eq_key)

@functools.cache
def _cached_solve(all_eq: EquationSet):
    """Solve equations with caching for performance.
    
    Parameters
    ----------
    all_eq : list or EquationSet
        Equations to solve.
    solve_vars : list[sym.Symbol]
        Variables to solve for.
    simplify : bool, optional
        Whether to simplify solutions (currently unused).
        
    Returns
    -------
    list[dict]
        List of unique solution dictionaries, with zero denominators filtered out.
        
    """
    solve_vars = all_eq.solve_vars
    eq_set_obj = all_eq

    if not any(v in eq.free_symbols for eq in eq_set_obj.as_eq_list() for v in solve_vars):
        return []

    # Recursive base case -- empty or single variable 1:1 mapping
    if len(eq_set_obj) == 0 or eq_set_obj._constant_conflict:
        return []
    elif eq_set_obj.as_dict() is not None and len(eq_set_obj) == 1:
        var, val = list(eq_set_obj.as_dict().items())[0]
        if var.is_symbol and var.is_number or var.is_symbol:
            return [{var: val}]
    
    # Put equations together and grab numerators
    eq_set_numer, all_denom = eq_set_obj.as_numer_denom()
    if _is_system_linear(eq_set_numer.as_eq_list(), solve_vars):
        general_sols = unique_solutions(_sols_set_to_dict(sym.linsolve(eq_set_numer.as_eq_list(), solve_vars), solve_vars))
    try:
        general_sols = unique_solutions(_sols_set_to_dict(sym.nonlinsolve(eq_set_numer.as_eq_list(), solve_vars), solve_vars))
    except Exception as e:
        try:
            general_sols = sym.solve(eq_set_numer.as_eq_list(), solve_vars, dict=True)
        except NotImplementedError:
            general_sols = []
    # Expand solutions by exploring singular branches
    sols = []
    for s in _expand_singular_branches(eq_set_numer, general_sols, solve_vars, nonzero=all_denom):
        if all(d.subs(s) != 0 for d in all_denom):
            sols.append(s)

    # Make sure none of the original denominators are zero
    sols = [s for s in sols if all(sym.simplify(d.subs(s)) != 0 for d in all_denom)]
    sols = unique_solutions(sols)
    sols = [{var: sym.simplify(val) for var, val in sol.items()} for sol in sols]

    return sols

def _is_system_linear(equations, variables):
    """Checks if all equations in a system are linear in all specified variables."""
    for eq in equations:
        # Rewrite the equation to the form expr = 0 for consistency
        expr = sym.expand(eq.lhs - eq.rhs) if eq.rhs != 0 else sym.expand(eq)
        for term in expr.args:
            var_in_term = [x for x in variables if x in term.free_symbols]
            if not var_in_term:
                continue
            elif len(var_in_term) > 1:
                return False
            elif not sym.Poly(term, var_in_term[0]).is_linear:
                return False
    return True


def _expand_singular_branches(eq_set, initial_solutions, solve_vars, nonzero=[]):
    """
    Expand solutions by exploring branches where denominators are zero.

    NOTE: By default it does not filter out solutions that make eq_set equations invalid.
    This is left to the caller to input appropriate nonzero conditions.

    Parameters
    ----------
    eq_set : EquationSet
        The original set of equations. 
    initial_solutions : list[dict]
        Initial solutions to expand upon. Assumed to be unique already.
    solve_vars : list[sym.Symbol]
        Variables to solve for. Assumes to be sorted already.
    """

    # Check for trivial case
    if eq_set == EquationSet.from_any(initial_solutions):
        return initial_solutions
    
    # Master list of all unique solutions found
    all_solutions = []
    
    # Queue for breadth first constraint traversal: stores (solution_dict, constraint_history_list)
    # We assume initial solutions have NO constraints (empty EquationSet)
    queue = [(sol, EquationSet.empty()) for sol in initial_solutions]
    
    # Loop Detection: Tracks sets of constraints we have already solved
    visited_constraints = set()
    visited_constraints.add(EquationSet.empty()) # Base case (no constraints) checked

    # Add the initial batch first
    for s in initial_solutions:
        all_solutions.append(s)

    # Expand each solution in the queue
    while queue:
        current_sol, current_constraints = queue.pop(0)
        
        # Extract all denominators from the current solution's values
        denominators = set()
        for val in current_sol.values():
            _, d = _eq_as_numer_denom(val)
            if any(v in d.free_symbols for v in solve_vars) and d not in nonzero:
                denominators.add(d)

        # Create New Branches
        for denom in denominators:
            # Create the new constraint: Denominator == 0
            new_constraint = sym.Eq(denom, 0)
            
            # Form the new state (Previous Constraints + New Constraint)
            next_constraints = current_constraints | EquationSet.from_any([new_constraint])

            # STOP: We have already solved this exact scenario or we have a contradiction
            if next_constraints in visited_constraints or next_constraints._constant_conflict:
                continue 
            
            visited_constraints.add(next_constraints)
            
            # Solve original equation + all accumulated constraints
            branch_system = eq_set | next_constraints
            new_sols = unique_solutions(_sols_set_to_dict(sym.nonlinsolve(branch_system.as_eq_list(), solve_vars), solve_vars))
            for s in new_sols:
                all_solutions.append(s)
                # Add to queue to check against further denominators
                queue.append((s, next_constraints))
                   

    return all_solutions


def _sols_set_to_dict(sols_set, solve_vars):
    """Convert SymPy nonlinsolve output to list of dictionaries.
    
    Parameters
    ----------
    sols_set : sympy solution set
        Output from sym.nonlinsolve.
    solve_vars : list[sym.Symbol]
        Variables corresponding to solution values.
        
    Returns
    -------
    list[dict]
        List of solution dictionaries, filtering out unbounded and trivial solutions.
    """
    sols = []
    for sol in sols_set:
        sol_dict = {}
        invalid = False
        for var, val in zip(solve_vars, sol):
            var, val = sym.simplify(var), sym.simplify(val)
            # If SymPy returns a set: allow Reals/Complexes (unconstrained), otherwise drop this solution
            if isinstance(val, sym.Set) and not isinstance(val, sym.Complement):
                if val in (sym.Reals, sym.Complexes):
                    continue
                if isinstance(val, sym.Interval):
                    raise ValueError("Unbounded solution encountered")
                invalid = True
                break

            if isinstance(val, sym.Complement):
                # Complement has structure: Complement(base_set, excluded_set)
                base_set = val.args[0]
                if {var} == base_set:
                    continue
                elif len(base_set) == 1:
                    base_elem = list(base_set)[0]
                    val = base_elem
                    continue
            # Variable can be any real/complex number; don't constrain it
            elif val == sym.Complexes or val == sym.Reals:
                continue
            elif (not isinstance(val, sym.Expr)) and (not isinstance(val, sym.Symbol)):
                print(var, val)
                breakpoint()
                invalid = True
                break
            if var != val:
                sol_dict[var] = val
        if not invalid:
            sols.append(sol_dict)
    return sols


def _eq_as_numer_denom(eq: Union[sym.Eq, sym.Expr]):
    """Extract numerator and denominator from equation or expression.
    
    Parameters
    ----------
    eq : sym.Eq or sym.Expr
        Equation or expression to decompose.
        
    Returns
    -------
    tuple[sym.Expr, sym.Expr]
        Numerator and denominator after combining fractions.
    """
    if isinstance(eq, sym.Eq):
        eq = (eq.lhs - eq.rhs) 
    eq_new = sym.expand(eq).together(deep=True)
    numer, denom = eq_new.as_numer_denom()
    return numer, denom


def unique_solutions(sols: Union[list[dict], list[EquationSet]], return_idx: bool = False) -> list[dict]:
    """Filter out duplicate and redundant solutions.
    
    Parameters
    ----------
    sols : list[dict]
        List of solution dictionaries.
        
    Returns
    -------
    list[dict]
        Filtered list with duplicates removed and minimal solutions kept.
        
    Notes
    -----
    Uses EquationSet for comparison so {x:1, y:2} and {y:2, x:1} are treated
    as identical. Keeps only minimal solutions (removes supersets).
    """
    if sols == []:
        if return_idx:
            return [], []
        else:
            return []
    if isinstance(sols[0], dict):
        sols = [EquationSet.from_any(sol) for sol in sols]
    kept: list[EquationSet] = []
    idx_kept = []
    og_idx = {sol: i for i, sol in enumerate(sols)}
    for sol in sorted(sols, key=lambda s: (len(s),og_idx[s])):
        # If any previously kept solution is a subset of this one, skip it
        if any(prev.issubset(sol) for prev in kept):
            continue
        kept.append(sol)
        idx_kept.append(og_idx[sol])
    if not return_idx:
        return [k.as_dict() for k in kept]
    return [k.as_dict() for k in kept], sorted(idx_kept)


def _unique_products(expr: sym.Expr, exclude: list[sym.Symbol] = []):
    """Extract unique symbolic products from expanded expression.
    
    Parameters
    ----------
    expr : sym.Expr
        SymPy expression to analyze.
    exclude : list[sym.Symbol], optional
        Symbols to exclude from products.
        
    Returns
    -------
    dict
        Mapping from product (sym.Expr) to coefficient (sym.Expr).
        
    Notes
    -----
    Expands expression and groups terms by unique products of symbols,
    excluding specified symbols from the products.
    """
    expr = sym.expand(expr)
    products = {}
    for term in expr.as_ordered_terms():
        # Extract multiplicative factors
        factors = []
        if isinstance(term, Mul):
            for f in term.args:
                if isinstance(f, sym.Symbol) and f not in exclude:
                    factors.append(f)
                elif isinstance(f, sym.Pow):
                    if f.base not in exclude:
                        factors.append(f)
        elif isinstance(term, sym.Pow):
            if term.base not in exclude:
                factors.append(term)
        elif isinstance(term, sym.Symbol) and term not in exclude:
            factors.append(term)
        elif term.is_number:
            factors.append(term)
        factors = tuple(sorted(factors, key=lambda a: str(a))) + (1,)
        prod = functools.reduce(lambda a,b: a*b, factors, sym.sympify(1))
        coeff = term/prod
        if factors in products:
            products[factors].append(coeff)
        else:
            products[factors] = [coeff]
            
    to_return = {}
    for factors, coeffs in products.items():
        prod = functools.reduce(lambda a,b: a*b, factors, sym.sympify(1))
        coeff = functools.reduce(lambda a,b: a+b, coeffs, sym.sympify(0))
        to_return[prod] = sym.simplify(coeff)
    
    return to_return


def extract_denom(Z: sym.Matrix):
    """Extract unique denominators from matrix elements.
    
    Parameters
    ----------
    Z : sym.Matrix
        SymPy matrix to analyze.
        
    Returns
    -------
    list[sym.Expr]
        List of unique denominators from non-constant matrix elements.
    """

    denom_list = []
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            if len(Z[i,j].free_symbols) > 0:
                denom_list.append(sym.simplify(Z[i,j].together().as_numer_denom()[1]))
    return list(set(denom_list))
