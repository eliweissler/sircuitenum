"""Equation set management and symbolic solving utilities.

This module provides the EquationSet class for managing collections of
symbolic equations with deduplication, caching, and compatibility checking.

Author: Eli Weissler
Version: 0.1.0
"""

__all__ = ["maximally_compatible_sol", "eq_indep_of_vars", "sol_indep_of_vars", "extract_denom"]

from dataclasses import dataclass, field
from typing import Union, Sequence, Iterable, Mapping, Optional, Tuple
import sympy as sym
import functools, itertools
from sympy.core.mul import Mul
import networkx as nx

from sircuitenum.singular_interface import solve_with_singular, extract_mappings, is_compatible, _eq_as_numer_denom


# Cache of solved equations
CACHE_VARS = sym.symbols(",".join([f'x{i}' for i in range(100)]), real=True)
CACHE_CONST = sym.symbols(",".join([f'c{i}' for i in range(100)]), real=True)
SOLVE_CACHE = {}
UNSOLVABLE_CACHE = set()  # sets of equations with no solutions


def maximally_compatible_sol(terms: list[list[sym.Expr]], nonzero: list[sym.Expr] = [],
                             nonzero_constraints: list[sym.Expr] = []) -> Tuple[list, list]:
    """
    Given a list of list of systems of equations, identifies the largest set of compatible
    systems of equations that can be solved simultaneously. Returns the indices of the selected
    systems along with the corresponding solutions.
    
    Assumes that any free symbols in the equations are to be solved for.
    
    Assumes that no fractions are present in the equations!

    Parameters
    ----------
    terms : list[list[sym.Expr]]
        List of term groups, each representing a system of equations
    nonzero : list[sym.Expr], optional
        Expressions that must be nonzero in solutions. Are checked after solving.
    nonzero_constraints : list[sym.Expr], optional
        Expressions to be included as constraints to the solver to enforce nonzero conditions.
        
    Returns
    -------
    tuple[list, list]
        - List of index tuples identifying best compatible term combinations
        - List of corresponding solution dictionaries

    """
    # Check for fractions in equations
    for term_group in terms:
        for eq in term_group:
            numer, denom = eq.as_numer_denom()
            if denom != 1:
                raise ValueError("Equations with fractions are not supported in maximally_compatible_sol.")
        
    # Rabinowitsch Trick to enforce nonzero conditions
    nz_term = []
    nz_var = sym.symbols('nzVar')
    if len(nonzero_constraints) > 0:
        nz_eq = sym.sympify(1)
        for d in nonzero_constraints:
            nz_eq *= d
        nz_term = [1 - nz_var * nz_eq]
        
    # Mark ones that are individually compatible
    n_terms = len(terms)
    is_solvable = [is_compatible(terms[i] + nz_term, check_fraction=False) if terms[i] else False for i in range(n_terms)]
    if sum(is_solvable) == 0:
        return [], []

    # Identify all incompatible pairs of terms
    # to speed up compatibility checking later
    incompatible_pairs = set()
    for i1, i2 in itertools.combinations(range(n_terms), 2):
        if not is_solvable[i1] or not is_solvable[i2]:
            incompatible_pairs.add((i1, i2))
            continue
        combined_terms = terms[i1] + terms[i2]
        if not is_compatible(combined_terms + nz_term, check_fraction=False):
            incompatible_pairs.add((i1, i2))
    

    G = nx.Graph()
    G.add_nodes_from(range(n_terms))
    G.add_edges_from(incompatible_pairs)

    # Top down search for largest compatible set
    all_sols = []
    sol_keys = []
    for nz in range(n_terms, 0, -1):
        sol_found = False
        for keys in itertools.combinations(range(n_terms), nz):
            # Check if all individual terms are solvable
            if not all(is_solvable[k] for k in keys):
                continue

            # Check if a subset of the keys are incompatible
            # by checking the conflict graph
            if G.subgraph(keys).number_of_edges() > 0:
                continue
            
            # Check compatibility of combined terms
            combined_eqs = list(itertools.chain.from_iterable(terms[k] for k in keys)) + nz_term
            if is_compatible(combined_eqs, check_fraction=False):
                # print("Solving combined eq", combined_eqs)
                # Solve combined equations
                sols = []
                branches = solve_with_singular(combined_eqs, check_fraction=False)
                for sol in extract_mappings(branches, real_only=True):
                    # Sub out nzvar if present
                    if nz_var in sol:
                        nz_var_val = sol.pop(nz_var)
                        sol = {k: v.subs(nz_var, nz_var_val) for k, v in sol.items()}
                    # Check nonzero conditions
                    if any(sym.simplify(d.subs(sol)) == 0 for d in nonzero):
                        continue
                    sols.append(sol)
                if len(sols) > 0:
                    all_sols.append(sols)
                    sol_keys.append(keys)
                    sol_found = True
        if sol_found:
            return sol_keys, all_sols
    
    # Should not reach here
    return [], []
    

def eq_indep_of_vars(expr: Union[sym.Eq, sym.Expr], exclude_vars: Iterable[sym.Symbol]):
    """
    Extract a system of equations whose solutions solve the input expression independently
    of the specified excluded variables.

    Args:
        expr (Union[sym.Eq, sym.Expr]): Equation or expression to analyze.
        exclude_vars: Iterable[sym.Symbol] (_type_): Variables you want solutions to be independent of.

    Returns:
        tuple[list[sym.Expr], sym.Expr]: The system of equations (as a list of expressions) whose solutions
        solve the input expression independently of the excluded variables, and the denominator
        of the original expression (i.e., this must be nonzero in the solutions).
    """
    
    # Simplify to rational form
    numer, denom = _eq_as_numer_denom(expr)

    # No solutions if there are no solve variables
    if not any(v in numer.free_symbols for v in exclude_vars):
        return [], sym.sympify(1)

    # Get unique symbolic products of non-excluded vars
    var_combos = _unique_products(numer, exclude=exclude_vars)

    # All coefficients must be able to be zero simultaneously
    coeffs = []
    # Constant term must be 0
    if 1 in var_combos:
        val = var_combos[1]
        if val.is_number and val != 0:
            return []
        else:
            coeffs.append(var_combos.pop(1))
    for v in var_combos:
        coeffs.append(var_combos[v])

    return coeffs, denom
    

def sol_indep_of_vars(expr: Union[sym.Eq, sym.Expr], solve_vars, nonzero=[],
                      real_only=True):
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
    
    coeffs, denom = eq_indep_of_vars(expr, solve_vars)
    sols = extract_mappings(solve_with_singular(coeffs, solve_vars), real_only=real_only)
    # breakpoint()
    # print("Found sols:", sols)

    # Make sure none of the nonzero conditions are violated
    sols = [s for s in sols if all(sym.simplify(d.subs(s)) != 0 for d in nonzero)]
    sols = [s for s in sols if sym.simplify(denom.subs(s)) != 0]

    return sols


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
        if term == 0 or term == 0.0:
            continue
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
        # to_return[prod] = sym.simplify(coeff)
        to_return[prod] = coeff
    
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
