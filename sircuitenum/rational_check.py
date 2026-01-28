import sympy as sym


def extract_numeric_part(expr, params):
    """
    Separates an expression into numeric and parametric parts.
    For a sum like: 4a² - 180a - 707
    Returns the constant term (numeric_part) and the rest.
    """
    expr = sym.simplify(expr).expand()
    
    if not params:
        return expr, sym.Integer(0)
    
    try:
        p = sym.Poly(expr, *params)
        # Constant term (no parameters)
        numeric_part = p.nth(*([0] * len(params)))
        parametric_part = expr - numeric_part
        return numeric_part, parametric_part
    except Exception:
        return None, expr


def is_perfect_power(n, k):
    """Check if integer n is a perfect k-th power."""
    if not isinstance(n, (int, sym.Integer)):
        try:
            n = int(n)
        except (TypeError, ValueError):
            return False
    
    if n == 0:
        return True
    
    if n < 0:
        if k % 2 == 0:
            return False
        n = abs(n)
    
    root = round(abs(n) ** (1/k))
    # Check root and neighbors due to floating point issues
    for r in [root - 1, root, root + 1]:
        if r > 0 and r ** k == abs(n):
            return True
    return False


def is_perfect_square(n):
    """Check if n is a perfect square."""
    return is_perfect_power(n, 2)


def is_perfect_cube(n):
    """Check if n is a perfect cube."""
    return is_perfect_power(n, 3)


def get_polynomial_params(poly):
    """Extract free symbols from polynomial coefficients (the parameters)."""
    params = set()
    for coeff in poly.all_coeffs():
        params.update(coeff.free_symbols)
    return list(params)


def is_binomial(poly):
    """
    Check if polynomial is a binomial of form x^n + c or x^n - c.
    Returns (True, n, c) if binomial, (False, None, None) otherwise.
    """
    # Get all terms with non-zero coefficients
    terms = poly.as_dict()
    
    if len(terms) != 2:
        return False, None, None
    
    n = sym.degree(poly)
    gen = poly.gen
    
    # Check for x^n term with coefficient 1
    if (n,) not in terms or terms[(n,)] != 1:
        return False, None, None
    
    # Check for constant term
    if (0,) not in terms:
        return False, None, None
    
    c = terms[(0,)]
    return True, n, c


def is_even_powers_only(poly):
    """
    Check if polynomial contains only even powers of x.
    Returns (True, reduced_poly) where reduced_poly is in terms of y = x².
    """
    terms = poly.as_dict()
    gen = poly.gen
    
    for (exp,) in terms.keys():
        if exp % 2 != 0:
            return False, None
    
    # Create reduced polynomial in y = x²
    y = sym.Symbol('_y_internal')
    new_expr = sum(coeff * y**(exp // 2) for (exp,), coeff in terms.items())
    reduced_poly = sym.Poly(new_expr, y)
    
    return True, reduced_poly


def check_binomial_rationality(n, c, params):
    """
    Check if x^n - c has clean roots.
    Clean if c is a perfect n-th power (for numeric c) or purely parametric.
    """
    if params:
        numeric, parametric = extract_numeric_part(c, params)
        if numeric is None:
            return None, f"Could not analyze binomial constant {c}"
        if parametric != 0:
            # Mixed numeric and parametric
            if numeric == 0:
                return True, "Binomial with purely parametric constant"
            # Check if numeric part is perfect n-th power
            is_perf = is_perfect_power(abs(numeric), n)
            return is_perf, f"Binomial x^{n} + {c}, numeric part {numeric} perfect {n}-th power: {is_perf}"
        else:
            # Pure numeric
            is_perf = is_perfect_power(abs(c), n)
            return is_perf, f"Binomial x^{n} + {c}, perfect {n}-th power: {is_perf}"
    else:
        # Pure numeric constant
        try:
            c_val = int(c)
            is_perf = is_perfect_power(abs(c_val), n)
            return is_perf, f"Binomial x^{n} + {c}, perfect {n}-th power: {is_perf}"
        except (TypeError, ValueError):
            return None, f"Could not evaluate binomial constant {c}"


def check_quadratic_rationality(poly):
    """
    For a quadratic polynomial, checks if roots are rational in parameters.
    
    Condition: The discriminant Δ = b² - 4ac must have its numeric
    (parameter-free) part be a perfect square (or zero).
    
    Parameters:
        poly: sympy Poly object, univariate quadratic
        
    Returns: (is_rational, discriminant, numeric_part, explanation)
    """
    if sym.degree(poly) != 2:
        raise ValueError(f"Expected degree 2, got {sym.degree(poly)}")
    
    # Check for binomial x² + c
    is_binom, n, c = is_binomial(poly)
    if is_binom:
        params = get_polynomial_params(poly)
        result, explanation = check_binomial_rationality(n, -c, params)
        return result, c, c, f"Binomial: {explanation}"
    
    coeffs = poly.all_coeffs()  # [a, b, c] for ax² + bx + c
    a, b, c = coeffs
    
    params = get_polynomial_params(poly)
    
    disc = sym.simplify((b**2 - 4*a*c).expand())
    
    numeric, parametric = extract_numeric_part(disc, params)
    
    if numeric is None:
        return None, disc, None, "Could not extract numeric part"
    
    if parametric == 0:
        is_rat = is_perfect_square(numeric)
        return is_rat, disc, numeric, f"Pure numeric: {numeric}, perfect square: {is_rat}"
    
    if numeric == 0:
        return True, disc, sym.Integer(0), "No numeric constant in discriminant"
    
    is_sq = is_perfect_square(abs(numeric))
    
    explanation = f"Numeric part: {numeric}, is perfect square: {is_sq}"
    if not is_sq:
        explanation += " → roots will contain irrational constants"
    
    return is_sq, disc, numeric, explanation


def check_cubic_rationality(poly):
    """
    For a cubic polynomial, checks if roots are rational in parameters.
    
    Checks Cardano's D = (q/2)² + (p/3)³
    
    Parameters:
        poly: sympy Poly object, univariate cubic
        
    Returns: (is_rational, D_expr, numeric_part, explanation)
    """
    if sym.degree(poly) != 3:
        raise ValueError(f"Expected degree 3, got {sym.degree(poly)}")
    
    # Check for binomial x³ + c
    is_binom, n, c = is_binomial(poly)
    if is_binom:
        params = get_polynomial_params(poly)
        result, explanation = check_binomial_rationality(n, -c, params)
        return result, c, c, f"Binomial: {explanation}"
    
    coeffs = poly.all_coeffs()  # [a, b, c, d] for ax³ + bx² + cx + d
    a, b, c, d = coeffs
    
    params = get_polynomial_params(poly)
    
    # Normalize to monic
    b, c, d = b/a, c/a, d/a
    
    # Depress: p and q for t³ + pt + q = 0
    p = sym.simplify(c - b**2 / 3)
    q = sym.simplify(d - b*c/3 + 2*b**3 / 27)
    
    # Cardano's D
    D = sym.simplify(((q/2)**2 + (p/3)**3).expand())
    
    # Multiply by 108 to clear common denominators
    D_scaled = sym.simplify((D * 108).expand())
    
    numeric, parametric = extract_numeric_part(D_scaled, params)
    
    if numeric is None:
        return None, D, None, "Could not extract numeric part"
    
    if parametric == 0:
        is_sq = is_perfect_square(abs(numeric))
        return is_sq, D, numeric, f"Pure numeric D*108: {numeric}"
    
    if numeric == 0:
        return True, D, sym.Integer(0), "No numeric constant in D → clean parametric roots"
    
    is_sq = is_perfect_square(abs(numeric))
    
    explanation = f"D*108 numeric part: {numeric}, is perfect square: {is_sq}"
    if not is_sq:
        explanation += " → roots will contain √(irrational)"
    
    return is_sq, D, numeric, explanation


def check_quartic_rationality(poly):
    """
    For a quartic polynomial, checks if roots are rational in parameters.
    
    Ferrari's method involves:
    1. A resolvent cubic
    2. Square roots of expressions involving the cubic's roots
    
    We check the resolvent cubic's discriminant and the quartic's discriminant.
    
    Also handles special cases:
    - Binomials x⁴ + c
    - Biquadratics (only even powers)
    
    Parameters:
        poly: sympy Poly object, univariate quartic
        
    Returns: (is_rational, discriminant, numeric_part, explanation)
    """
    if sym.degree(poly) != 4:
        raise ValueError(f"Expected degree 4, got {sym.degree(poly)}")
    
    params = get_polynomial_params(poly)
    
    # Check for binomial x⁴ + c
    is_binom, n, c = is_binomial(poly)
    if is_binom:
        result, explanation = check_binomial_rationality(n, -c, params)
        return result, c, c, f"Binomial: {explanation}"
    
    # Check for biquadratic (only even powers)
    is_even, reduced_poly = is_even_powers_only(poly)
    if is_even:
        # Reduce to quadratic in y = x²
        reduced_degree = sym.degree(reduced_poly)
        if reduced_degree == 2:
            quad_result = check_quadratic_rationality(reduced_poly)
            return quad_result[0], quad_result[1], quad_result[2], f"Biquadratic reduces to quadratic: {quad_result[3]}"
        elif reduced_degree == 1:
            return True, None, None, "Biquadratic reduces to linear"
    
    coeffs = poly.all_coeffs()
    a, b, c, d, e = coeffs  # ax⁴ + bx³ + cx² + dx + e
    
    # Normalize to monic
    b, c, d, e = b/a, c/a, d/a, e/a
    
    # Resolvent cubic: y³ - cy² + (bd - 4e)y - (b²e - 4ce + d²)
    gen = poly.gen
    rc_a = sym.Integer(1)
    rc_b = -c
    rc_c = b*d - 4*e
    rc_d = -(b**2 * e - 4*c*e + d**2)
    
    resolvent = sym.Poly(
        rc_a * gen**3 + rc_b * gen**2 + rc_c * gen + rc_d,
        gen
    )
    
    cubic_result = check_cubic_rationality(resolvent)
    
    # Also compute quartic discriminant's numeric part
    disc = sym.simplify((
        256*e**3 - 192*b*d*e**2 - 128*c**2*e**2 + 144*c*d**2*e 
        - 27*d**4 + 144*b*c**2*e - 6*b**2*d**2*e - 80*b*c*d**2 
        + 18*b*c*d**3 + 16*c**4 - 4*c**3*d**2 - 27*b**4*e**2 
        + 18*b**3*d*e - 4*b**2*c**3 + b**2*c**2*d**2
    ).expand())
    
    numeric, parametric = extract_numeric_part(disc, params)
    
    if numeric is None:
        return cubic_result[0], disc, None, f"Resolvent cubic check: {cubic_result[3]}"
    
    if parametric == 0:
        is_sq = is_perfect_square(abs(numeric))
        return is_sq and cubic_result[0], disc, numeric, f"Pure numeric, resolvent: {cubic_result[0]}"
    
    if numeric == 0:
        return cubic_result[0], disc, sym.Integer(0), f"No numeric constant, resolvent: {cubic_result[3]}"
    
    is_sq = is_perfect_square(abs(numeric))
    combined = is_sq and cubic_result[0]
    
    explanation = f"Disc numeric: {numeric}, perfect sq: {is_sq}, resolvent: {cubic_result[0]}"
    
    return combined, disc, numeric, explanation


def check_higher_degree_rationality(poly):
    """
    For degree ≥ 5, we use a heuristic based on the discriminant.
    
    The discriminant appears under radicals in any radical solution
    (when one exists). If the numeric part isn't a perfect square,
    the roots will contain irrational constants.
    
    Also handles special cases:
    - Binomials x^n + c
    - Polynomials with only even powers
    
    This is a necessary but NOT sufficient condition.
    (Degree ≥ 5 polynomials may not even be solvable by radicals!)
    
    Parameters:
        poly: sympy Poly object
        
    Returns: (passes_filter, discriminant, numeric_part, explanation)
    """
    n = sym.degree(poly)
    params = get_polynomial_params(poly)
    
    # Check for binomial x^n + c
    is_binom, deg, c = is_binomial(poly)
    if is_binom:
        result, explanation = check_binomial_rationality(deg, -c, params)
        return result, c, c, f"Binomial: {explanation}"
    
    # Check for even powers only (reduces degree)
    is_even, reduced_poly = is_even_powers_only(poly)
    if is_even:
        reduced_degree = sym.degree(reduced_poly)
        if reduced_degree <= 4:
            reduced_result, reduced_details = check_rationality(reduced_poly)
            return reduced_result, None, None, f"Even powers only, reduces to degree {reduced_degree}: {reduced_details.get('explanation', '')}"
        else:
            # Recursively check the reduced polynomial
            reduced_result = check_higher_degree_rationality(reduced_poly)
            return reduced_result[0], reduced_result[1], reduced_result[2], f"Even powers only, reduces to degree {reduced_degree}: {reduced_result[3]}"
    
    try:
        disc = sym.simplify(sym.discriminant(poly).expand())
    except Exception:
        return None, None, None, "Could not compute discriminant"
    
    numeric, parametric = extract_numeric_part(disc, params)
    
    if numeric is None:
        return None, disc, None, "Could not extract numeric part"
    
    if parametric == 0:
        is_sq = is_perfect_square(abs(numeric))
        return is_sq, disc, numeric, f"Pure numeric discriminant, perfect square: {is_sq}"
    
    if numeric == 0:
        return True, disc, sym.Integer(0), "No numeric constant in discriminant (passes filter)"
    
    is_sq = is_perfect_square(abs(numeric))
    
    explanation = f"Degree {n}: Disc numeric part: {numeric}, perfect square: {is_sq}"
    if not is_sq:
        explanation += " → likely contains irrational constants"
    else:
        explanation += " → passes filter (but may still have irrationals)"
    
    return is_sq, disc, numeric, explanation


def check_rationality(poly):
    """
    Main entry point: checks if polynomial roots are rational in parameters.
    
    Parameters:
        poly: sympy Poly object (univariate)
        
    Returns: (passes_check, details_dict)
        passes_check: True if roots likely contain only rational expressions
                      in (roots of) parameters
        details_dict: Dictionary with discriminant, numeric_part, explanation
    """
    if not isinstance(poly, sym.Poly):
        raise TypeError("Expected a sympy Poly object")
    
    n = sym.degree(poly)
    
    if n == 0:
        return True, {"explanation": "Constant polynomial - no roots"}
    
    if n == 1:
        return True, {"explanation": "Linear polynomial - root is rational"}
    
    if n == 2:
        result, disc, numeric, explanation = check_quadratic_rationality(poly)
        return result, {
            "discriminant": disc,
            "numeric_part": numeric,
            "explanation": explanation
        }
    
    if n == 3:
        result, D, numeric, explanation = check_cubic_rationality(poly)
        return result, {
            "cardano_D": D,
            "numeric_part": numeric,
            "explanation": explanation
        }
    
    if n == 4:
        result, disc, numeric, explanation = check_quartic_rationality(poly)
        return result, {
            "discriminant": disc,
            "numeric_part": numeric,
            "explanation": explanation
        }
    
    # Degree 5+
    result, disc, numeric, explanation = check_higher_degree_rationality(poly)
    return result, {
        "discriminant": disc,
        "numeric_part": numeric,
        "explanation": explanation,
        "warning": "Degree ≥ 5: polynomial may not be solvable by radicals at all"
    }


def has_clean_roots(poly):
    """Simple boolean check - does the polynomial likely have clean roots?"""
    result, _ = check_rationality(poly)
    return result


def has_at_least_one_clean_root(poly):
    """
    Check if the polynomial has at least one "clean" root.
    
    A clean root is one expressible as a rational function of 
    (roots of) the parameters, without irrational numeric constants.
    
    Strategy:
    1. Try to factor the polynomial
    2. Check if any factor is linear (gives rational root)
    3. Check if any factor passes the all-roots-clean test
    
    Parameters:
        poly: sympy Poly object (univariate)
        
    Returns: (has_clean_root, details_dict)
    """
    if not isinstance(poly, sym.Poly):
        raise TypeError("Expected a sympy Poly object")
    
    n = sym.degree(poly)
    
    if n == 0:
        return False, {"explanation": "Constant polynomial - no roots"}
    
    if n == 1:
        return True, {"explanation": "Linear polynomial - root is rational"}
    
    params = get_polynomial_params(poly)
    gen = poly.gen
    
    # Try to factor the polynomial
    try:
        factors = sym.factor_list(poly.as_expr(), gen)
        # factors is (content, [(factor1, mult1), (factor2, mult2), ...])
        content, factor_list = factors
        
        clean_factors = []
        
        for factor_expr, multiplicity in factor_list:
            factor_poly = sym.Poly(factor_expr, gen)
            factor_degree = sym.degree(factor_poly)
            
            # Linear factors always give clean roots
            if factor_degree == 1:
                # Extract the root from ax + b → x = -b/a
                coeffs = factor_poly.all_coeffs()
                a, b = coeffs
                root = -b / a
                clean_factors.append({
                    "factor": factor_expr,
                    "degree": 1,
                    "root": root,
                    "clean": True,
                    "multiplicity": multiplicity
                })
                continue
            
            # For higher degree factors, check if all roots are clean
            result, details = check_rationality(factor_poly)
            clean_factors.append({
                "factor": factor_expr,
                "degree": factor_degree,
                "clean": result,
                "details": details,
                "multiplicity": multiplicity
            })
        
        # Check if any factor has clean roots
        has_clean = any(f["clean"] for f in clean_factors)
        
        return has_clean, {
            "explanation": "Factored polynomial",
            "factors": clean_factors,
            "num_clean_factors": sum(1 for f in clean_factors if f["clean"])
        }
        
    except Exception as e:
        # Factoring failed, fall back to checking all roots
        result, details = check_rationality(poly)
        return result, {
            "explanation": f"Could not factor, checked all roots: {details.get('explanation', '')}",
            "factoring_error": str(e)
        }


def get_clean_roots(poly):
    """
    Get the clean roots of a polynomial (if any).
    
    Returns roots that are expressible as rational functions of
    (roots of) the parameters.
    
    Parameters:
        poly: sympy Poly object (univariate)
        
    Returns: list of clean roots (may be empty)
    """
    if not isinstance(poly, sym.Poly):
        raise TypeError("Expected a sympy Poly object")
    
    n = sym.degree(poly)
    gen = poly.gen
    
    if n == 0:
        return []
    
    if n == 1:
        coeffs = poly.all_coeffs()
        a, b = coeffs
        return [sym.simplify(-b / a)]
    
    clean_roots = []
    
    # Try to factor the polynomial
    try:
        factors = sym.factor_list(poly.as_expr(), gen)
        content, factor_list = factors
        
        for factor_expr, multiplicity in factor_list:
            factor_poly = sym.Poly(factor_expr, gen)
            factor_degree = sym.degree(factor_poly)
            
            # Linear factors give clean roots directly
            if factor_degree == 1:
                coeffs = factor_poly.all_coeffs()
                a, b = coeffs
                root = sym.simplify(-b / a)
                clean_roots.extend([root] * multiplicity)
                continue
            
            # For higher degree factors, check if all roots are clean
            result, _ = check_rationality(factor_poly)
            if result:
                # All roots of this factor are clean, solve it
                try:
                    roots = sym.solve(factor_expr, gen)
                    clean_roots.extend(roots * multiplicity)
                except Exception:
                    pass  # Couldn't solve, skip
        
        return clean_roots
        
    except Exception:
        # Factoring failed, try solving directly if all roots are clean
        result, _ = check_rationality(poly)
        if result:
            try:
                return sym.solve(poly.as_expr(), gen)
            except Exception:
                return []
        return []


def count_clean_roots(poly):
    """
    Count how many clean roots the polynomial has (with multiplicity).
    
    Parameters:
        poly: sympy Poly object (univariate)
        
    Returns: int
    """
    return len(get_clean_roots(poly))


def _has_rational_coeffs(expr, params):
    """
    Validates that 'expr' contains ONLY Rational numbers as coefficients.
    
    Allowed:
      - 1/2 * a
      - sqrt(a)          (Pow(a, 1/2) -> Base has params -> OK)
      - (a + b)^(1/3)    (Base has params -> OK)
      
    Rejected:
      - sqrt(2) * a      (Pow(2, 1/2) -> Base is Number, not perfect square -> REJECT)
      - pi * a           (Transcendental number -> REJECT)
      - 1.414 * a        (Float -> REJECT, unless nsimplified first)
    """
    # Float Check
    if expr.has(sym.Float):
        return False
        
    # Rejects Pi, E, any trig
    if expr.has(sym.pi, sym.E, sym.sin, sym.cos, sym.tan):
        return False

    # Look for "Bad Powers" (sqrt(2), 5^(1/3))
    # We look for Pow(b, e) where b is a Number.
    for node in sym.preorder_traversal(expr):
        if node.is_Pow:
            node = sym.nsimplify(node)
            base, exp = node.args
            # Case A: Base is a Number (e.g., 2^(1/2))
            if base.is_Number:
                # If exponent is Integer, it's Rational (2^3 = 8). OK.
                if not exp.is_Integer:
                    return False
                
    return True
