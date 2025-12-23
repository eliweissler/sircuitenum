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
    _pair_key: frozenset = field(init=False, repr=False)
    _dict: Optional[dict] = field(init=False, repr=False, default=None)
    _canonical_key: frozenset = field(init=False, repr=False)
    _constant_conflict: bool = field(init=False, repr=True, default=False)
    _solve_vars: bool = field(init=False, repr=True, default=False)

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
            The variables to solve for (required for canonicalization).
            
        Returns
        -------
        EquationSet
            New EquationSet with normalized equations.
        """
        eqs = cls._coerce(raw)
        # normalize by removing trivially true eqs; keep ordering stable
        norm = tuple(eq for eq in eqs if not (eq.lhs.is_number and eq.rhs.is_number and eq.lhs == eq.rhs))
        obj = cls(norm)
        object.__setattr__(obj, "_solve_vars", solve_vars)
        return obj
    
    @classmethod
    def empty(cls) -> "EquationSet":
        """Create an empty EquationSet.
        
        Returns
        -------
        EquationSet
            Empty equation set with no equations.
        """
        return cls(tuple())

    def __post_init__(self):
        object.__setattr__(self, "_pair_key", frozenset(self._pair(eq) for eq in self.eqs))
        object.__setattr__(self, "_dict", self._build_dict())
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
        for eq in self.eqs:
            var, val = eq.lhs, eq.rhs
            # Number != Number
            if var.is_number and val.is_number:
                if var != val:
                    object.__setattr__(self, "_constant_conflict", True)
                    return None
                continue
            # val is number and var is not -> swap
            if var.is_number and not val.is_number:
                var, val = val, var
            # Variable already assigned a different number
            if not var.is_number and val.is_number:
                if var in d and d[var].is_number and d[var] != val:
                    object.__setattr__(self, "_constant_conflict", True)
                    return None
            if var in d and d[var] != val:
                return None
            d[var] = val
        return d
    
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
        return self._pair_key

    def as_eq_list(self) -> list[sym.Eq]:
        """Convert to list of SymPy equations.
        
        Returns
        -------
        list[sym.Eq]
            List of equations in this set.
        """
        return list(self.eqs)

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
        return EquationSet.from_any(self.eqs + other.eqs)

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
        for eq in self.eqs:
            numer, denom = _eq_as_numer_denom(eq)
            denoms.append(denom)
            numer_eqs.append(sym.Eq(numer, 0))
        return EquationSet.from_any(numer_eqs), denoms


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
            if nz > n_terms - max(len(incompat_keys) for incompat_keys in incompatible):
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
            good_sub_sets = set()
            tiebreak_val = tiebreaker_fn(keys)
            for i, compat_sub in enumerate(res):
                compat_eq = EquationSet.from_any(compat_sub, solve_vars=solve_vars)
                if compat_eq._constant_conflict:
                    continue
                if compat_eq in good_sub_sets:
                    continue
                if any(sym.simplify(d.subs(compat_sub)) == 0 for d in nonzero):
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
                good_sub_sets.add(compat_eq)
            # Save the results for these keys
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

    if (starting, assumptions, frozenset(nonzero)) in COMPATIBLE_CACHE:
        return COMPATIBLE_CACHE[(starting, assumptions, frozenset(nonzero))]

    # Starting is not a valid set of substitutions
    if starting._constant_conflict:
        return []
    
    # Base case: no more assumptions to process
    if len(assumptions) == 0:
        return [starting.as_dict()]
    
    # Record fixed substitutions for the starting assumptions
    const_subs = {var: val for var, val in starting.as_dict().items() if val.is_number and isinstance(var, sym.Symbol)}

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
        sols = cached_solve(combined_eq_set.as_eq_list(), solve_vars)
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
        COMPATIBLE_CACHE[(starting, assumptions)] = all_res

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
    sols = []
    for coeff in coeffs:
        si = cached_solve([coeff], solve_vars)
        if si:
            sols.append(si)
        # At least one of them is unsolvable
        else:
            return []
    sols = unique_solutions(fully_compatible_set(sols, solve_vars, nonzero=[]))

    # Make sure none of the nonzero conditions are violated
    sols = [s for s in sols if all(sym.simplify(d.subs(s)) != 0 for d in nonzero)]
    sols = [s for s in sols if sym.simplify(denom.subs(s)) != 0]

    # Verify that the original expression is indeed zero
    if any(not sym.simplify(expr.subs(s)).is_zero for s in sols):
        raise ValueError("Solution does not satisfy original expression")

    return sols


def cached_solve(all_eq: Union[EquationSet, list], solve_vars: list[sym.Symbol],
                 simplify: bool = True):
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
        
    Notes
    -----
    Uses SOLVE_CACHE and UNSOLVABLE_CACHE for memoization.
    """

    # Prevent undefined behavior from solve variable order
    solve_vars = sorted(solve_vars, key=lambda x: str(x))

    # Convert to EquationSet
    if not isinstance(all_eq, EquationSet):
        eq_set_obj = EquationSet.from_any(all_eq, solve_vars=solve_vars)
    else:
        eq_set_obj = all_eq

    # Recursive base case
    if len(eq_set_obj) == 0:
        return []
    
    # Check if we've solved before
    if eq_set_obj in UNSOLVABLE_CACHE:
        return []
    elif any(unsolv.issubset(eq_set_obj) for unsolv in UNSOLVABLE_CACHE):
        UNSOLVABLE_CACHE.add(eq_set_obj)
        return []
    if eq_set_obj in SOLVE_CACHE:
        return SOLVE_CACHE[eq_set_obj]
    
    # Put equations together and grab numerators
    eq_set_numer, all_denom = eq_set_obj.as_numer_denom()
    # Check if this exact set or any subset is known to be unsolvable
    if eq_set_numer in UNSOLVABLE_CACHE:
        return []
    elif any(unsolv.issubset(eq_set_numer) for unsolv in UNSOLVABLE_CACHE):
        UNSOLVABLE_CACHE.add(eq_set_numer)
        return []
    # Have we solved this set of equations before?
    if eq_set_numer in SOLVE_CACHE:
        sols = SOLVE_CACHE[eq_set_numer]
    else:
        # Make solution list of dictionaries
        sols = unique_solutions(_sols_set_to_dict(sym.nonlinsolve(eq_set_numer.as_eq_list(), solve_vars), solve_vars))
        if sols:
            SOLVE_CACHE[eq_set_numer] = sols
        else:
            UNSOLVABLE_CACHE.add(eq_set_numer)
            return []

    # Make sure none of the original denominators are zero
    sols = [s for s in sols if all(sym.simplify(d.subs(s)) != 0 for d in all_denom)]

    # Cache the results
    if sols:
        SOLVE_CACHE[eq_set_obj] = sols
    else:
        UNSOLVABLE_CACHE.add(eq_set_obj)
    
    return sols


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
        for var, val in zip(solve_vars, sol):
            var, val = sym.simplify(var), sym.simplify(val)
            if isinstance(val, sym.Complement):
                if {var} in val.args:
                    continue
            if val == sym.Complexes or val == sym.Reals:
                continue
            if var != val:
                sol_dict[var] = val
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
