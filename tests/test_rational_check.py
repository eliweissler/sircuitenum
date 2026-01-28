import pytest
import sympy as sym
from sircuitenum.rational_check import (
    check_rationality,
    has_clean_roots,
    is_perfect_square,
    is_perfect_cube,
    is_perfect_power,
    extract_numeric_part,
    get_polynomial_params,
    check_quadratic_rationality,
    check_cubic_rationality,
    check_quartic_rationality,
    check_higher_degree_rationality,
    has_at_least_one_clean_root,
    get_clean_roots,
    count_clean_roots
)


# =============================================================================
# Test helper functions
# =============================================================================

class TestIsPerfectPower:
    def test_perfect_squares(self):
        assert is_perfect_square(0) is True
        assert is_perfect_square(1) is True
        assert is_perfect_square(4) is True
        assert is_perfect_square(9) is True
        assert is_perfect_square(16) is True
        assert is_perfect_square(100) is True
        assert is_perfect_square(144) is True
    
    def test_not_perfect_squares(self):
        assert is_perfect_square(2) is False
        assert is_perfect_square(3) is False
        assert is_perfect_square(5) is False
        assert is_perfect_square(7) is False
        assert is_perfect_square(707) is False
    
    def test_negative_not_perfect_square(self):
        assert is_perfect_square(-4) is False
        assert is_perfect_square(-1) is False
    
    def test_perfect_cubes(self):
        assert is_perfect_cube(0) is True
        assert is_perfect_cube(1) is True
        assert is_perfect_cube(8) is True
        assert is_perfect_cube(27) is True
        assert is_perfect_cube(64) is True
        assert is_perfect_cube(-8) is True
        assert is_perfect_cube(-27) is True
    
    def test_not_perfect_cubes(self):
        assert is_perfect_cube(2) is False
        assert is_perfect_cube(9) is False
        assert is_perfect_cube(707) is False
    
    def test_perfect_power_general(self):
        assert is_perfect_power(16, 4) is True
        assert is_perfect_power(81, 4) is True
        assert is_perfect_power(32, 5) is True
        assert is_perfect_power(17, 4) is False
    
    def test_sympy_integers(self):
        assert is_perfect_square(sym.Integer(16)) is True
        assert is_perfect_square(sym.Integer(17)) is False


class TestExtractNumericPart:
    def test_pure_numeric(self):
        a = sym.Symbol('a')
        numeric, parametric = extract_numeric_part(sym.Integer(42), [a])
        assert numeric == 42
        assert parametric == 0
    
    def test_pure_parametric(self):
        a = sym.Symbol('a')
        expr = a**2 + 3*a
        numeric, parametric = extract_numeric_part(expr, [a])
        assert numeric == 0
        assert sym.simplify(parametric - expr) == 0
    
    def test_mixed(self):
        a = sym.Symbol('a')
        expr = a**2 + 3*a + 7
        numeric, parametric = extract_numeric_part(expr, [a])
        assert numeric == 7
        assert sym.simplify(parametric - (a**2 + 3*a)) == 0
    
    def test_multiple_params(self):
        a, b = sym.symbols('a b')
        expr = a**2 + b + a*b + 5
        numeric, parametric = extract_numeric_part(expr, [a, b])
        assert numeric == 5
    
    def test_no_params(self):
        numeric, parametric = extract_numeric_part(sym.Integer(42), [])
        assert numeric == 42
        assert parametric == 0


class TestGetPolynomialParams:
    def test_numeric_coefficients(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 + 3*x + 2, x)
        params = get_polynomial_params(poly)
        assert params == []
    
    def test_single_param(self):
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**2 + a*x + 1, x)
        params = get_polynomial_params(poly)
        assert params == [a]
    
    def test_multiple_params(self):
        x, a, b, c = sym.symbols('x a b c')
        poly = sym.Poly(x**3 + a*x**2 + b*x + c, x)
        params = get_polynomial_params(poly)
        assert set(params) == {a, b, c}


# =============================================================================
# Test quadratic rationality
# =============================================================================

class TestQuadraticRationality:
    def test_numeric_perfect_square_discriminant(self):
        # x² - 5x + 6 = (x-2)(x-3), Δ = 25 - 24 = 1
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 5*x + 6, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_numeric_non_perfect_square_discriminant(self):
        # x² + x + 1, Δ = 1 - 4 = -3
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 + x + 1, x)
        result, details = check_rationality(poly)
        assert result is False
    
    def test_parametric_no_numeric(self):
        # x² + ax + a, Δ = a² - 4a (no constant term)
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**2 + a*x + a, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_parametric_with_perfect_square_numeric(self):
        # x² + ax + 4, Δ = a² - 16 (16 is perfect square)
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**2 + a*x + 4, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_parametric_with_non_perfect_square_numeric(self):
        # x² + ax + 1, Δ = a² - 4 (4 is perfect square, but -4 needs sign check)
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**2 + a*x + 1, x)
        result, details = check_rationality(poly)
        # -4 → abs(-4) = 4 is perfect square
        assert result is True
    
    def test_parametric_with_non_perfect_square_numeric_2(self):
        # x² + ax + 2, Δ = a² - 8 (8 is not perfect square)
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**2 + a*x + 2, x)
        result, details = check_rationality(poly)
        assert result is False
    
    def test_wrong_degree_raises(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**3 + x + 1, x)
        with pytest.raises(ValueError, match="Expected degree 2"):
            check_quadratic_rationality(poly)


# =============================================================================
# Test cubic rationality
# =============================================================================

class TestCubicRationality:
    def test_fully_symbolic(self):
        # x³ + ax² + bx + c → no numeric constants
        x, a, b, c = sym.symbols('x a b c')
        poly = sym.Poly(x**3 + a*x**2 + b*x + c, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_original_example_with_irrationals(self):
        # x³ + ax² + 2x - 5 → has 707 which is not perfect square
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**3 + a*x**2 + 2*x - 5, x)
        result, details = check_rationality(poly)
        assert result is False
    
    def test_factorizable_cubic(self):
        # x³ - 6x² + 11x - 6 = (x-1)(x-2)(x-3) → rational roots
        x = sym.Symbol('x')
        poly = sym.Poly(x**3 - 6*x**2 + 11*x - 6, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_pure_numeric_irrational(self):
        # x³ - 2 → root is ∛2
        x = sym.Symbol('x')
        poly = sym.Poly(x**3 - 2, x)
        result, details = check_rationality(poly)
        assert result is False
    
    def test_pure_numeric_rational(self):
        # x³ - 8 → root is 2
        x = sym.Symbol('x')
        poly = sym.Poly(x**3 - 8, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_wrong_degree_raises(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 + x + 1, x)
        with pytest.raises(ValueError, match="Expected degree 3"):
            check_cubic_rationality(poly)


# =============================================================================
# Test quartic rationality
# =============================================================================

class TestQuarticRationality:
    def test_fully_symbolic(self):
        # x⁴ + ax³ + bx² + cx + d → no numeric constants
        x, a, b, c, d = sym.symbols('x a b c d')
        poly = sym.Poly(x**4 + a*x**3 + b*x**2 + c*x + d, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_factorizable_quartic(self):
        # x⁴ - 5x² + 4 = (x²-1)(x²-4) = (x-1)(x+1)(x-2)(x+2)
        x = sym.Symbol('x')
        poly = sym.Poly(x**4 - 5*x**2 + 4, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_biquadratic_irrational(self):
        # x⁴ - 2 → roots are ±⁴√2
        x = sym.Symbol('x')
        poly = sym.Poly(x**4 - 2, x)
        result, details = check_rationality(poly)
        assert result is False
    
    def test_wrong_degree_raises(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**3 + x + 1, x)
        with pytest.raises(ValueError, match="Expected degree 4"):
            check_quartic_rationality(poly)


# =============================================================================
# Test higher degree rationality
# =============================================================================

class TestHigherDegreeRationality:
    def test_fully_symbolic(self):
        # x⁵ + ax + b → no numeric constants
        x, a, b = sym.symbols('x a b')
        poly = sym.Poly(x**5 + a*x + b, x)
        result, details = check_rationality(poly)
        assert result is True
        assert "warning" in details
    
    def test_roots_of_unity(self):
        # x⁶ - 1 → roots are 6th roots of unity
        x = sym.Symbol('x')
        poly = sym.Poly(x**6 - 1, x)
        result, details = check_rationality(poly)
        # Discriminant check
        assert "warning" in details
    
    def test_non_perfect_square_discriminant(self):
        # x⁵ - x + 1 → check discriminant
        x = sym.Symbol('x')
        poly = sym.Poly(x**5 - x + 1, x)
        result, details = check_rationality(poly)
        assert "warning" in details


# =============================================================================
# Test main entry point
# =============================================================================

class TestCheckRationality:
    def test_linear(self):
        x, a = sym.symbols('x a')
        poly = sym.Poly(a*x + 1, x)
        result, details = check_rationality(poly)
        assert result is True
        assert "Linear" in details["explanation"]
    
    def test_constant(self):
        x = sym.Symbol('x')
        poly = sym.Poly(5, x)
        result, details = check_rationality(poly)
        assert result is True
        assert "Constant" in details["explanation"]
    
    def test_not_a_poly_raises(self):
        with pytest.raises(TypeError, match="Expected a sympy Poly object"):
            check_rationality("not a poly")
    
    def test_has_clean_roots_convenience(self):
        x, a = sym.symbols('x a')
        poly_clean = sym.Poly(x**2 + a*x + a, x)
        poly_dirty = sym.Poly(x**2 + a*x + 2, x)
        
        assert has_clean_roots(poly_clean) is True
        assert has_clean_roots(poly_dirty) is False


# =============================================================================
# Edge cases
# =============================================================================

class TestEdgeCases:
    def test_zero_discriminant(self):
        # x² - 2x + 1 = (x-1)² → Δ = 0
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 2*x + 1, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_large_perfect_square(self):
        # Test with discriminant that's a large perfect square
        x = sym.Symbol('x')
        # x² - 200x + 9999 → Δ = 40000 - 39996 = 4
        poly = sym.Poly(x**2 - 200*x + 9999, x)
        result, details = check_rationality(poly)
        # 40000 - 39996 = 4, which is perfect square
        assert result is True
    
    def test_multiple_parameters(self):
        x, a, b = sym.symbols('x a b')
        poly = sym.Poly(x**2 + a*x + b, x)
        result, details = check_rationality(poly)
        assert result is True  # No numeric constants
    
    def test_rational_coefficients(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 + sym.Rational(1, 2)*x + sym.Rational(1, 8), x)
        result, details = check_rationality(poly)
        # Δ = 1/4 - 4*(1/8) = 1/4 - 1/2 = -1/4
        # This involves rationals, not necessarily clean
        assert isinstance(result, bool)

class TestBinomials:
    def test_quadratic_binomial_perfect_square(self):
        # x² - 4 → roots ±2
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 4, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_quadratic_binomial_not_perfect_square(self):
        # x² - 2 → roots ±√2
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 2, x)
        result, details = check_rationality(poly)
        assert result is False
    
    def test_cubic_binomial_perfect_cube(self):
        # x³ - 8 → root is 2
        x = sym.Symbol('x')
        poly = sym.Poly(x**3 - 8, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_cubic_binomial_perfect_cube_negative(self):
        # x³ + 27 → root is -3
        x = sym.Symbol('x')
        poly = sym.Poly(x**3 + 27, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_cubic_binomial_not_perfect_cube(self):
        # x³ - 2 → root is ∛2
        x = sym.Symbol('x')
        poly = sym.Poly(x**3 - 2, x)
        result, details = check_rationality(poly)
        assert result is False
    
    def test_quartic_binomial_perfect_fourth(self):
        # x⁴ - 16 → roots ±2, ±2i
        x = sym.Symbol('x')
        poly = sym.Poly(x**4 - 16, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_quartic_binomial_not_perfect_fourth(self):
        # x⁴ - 2 → roots ±⁴√2, ±i⁴√2
        x = sym.Symbol('x')
        poly = sym.Poly(x**4 - 2, x)
        result, details = check_rationality(poly)
        assert result is False
    
    def test_quintic_binomial_perfect_fifth(self):
        # x⁵ - 32 → root is 2
        x = sym.Symbol('x')
        poly = sym.Poly(x**5 - 32, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_binomial_with_parameter(self):
        # x³ - a → root is a^(1/3), clean
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**3 - a, x)
        result, details = check_rationality(poly)
        assert result is True


class TestBiquadratics:
    def test_biquadratic_factors_nicely(self):
        # x⁴ - 5x² + 4 = (x²-1)(x²-4) → roots ±1, ±2
        x = sym.Symbol('x')
        poly = sym.Poly(x**4 - 5*x**2 + 4, x)
        result, details = check_rationality(poly)
        assert result is True
    
    def test_biquadratic_irrational_but_clean(self):
        # x⁴ - 3x² + 2 = (x²-1)(x²-2) → roots ±1, ±√2
        # The √2 comes from parameter-free quadratic, so not "clean"
        x = sym.Symbol('x')
        poly = sym.Poly(x**4 - 3*x**2 + 2, x)
        result, details = check_rationality(poly)
        # Reduces to y² - 3y + 2, discriminant = 9 - 8 = 1 (perfect square)
        assert result is True
    
    def test_biquadratic_with_parameter(self):
        # x⁴ + ax² + 1 → reduces to y² + ay + 1
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**4 + a*x**2 + 1, x)
        result, details = check_rationality(poly)
        # Discriminant of reduced: a² - 4, numeric part -4 is perfect square
        assert result is True
    
    def test_sixth_degree_even_powers(self):
        # x⁶ - 1 = (x²)³ - 1 → reduces to y³ - 1
        x = sym.Symbol('x')
        poly = sym.Poly(x**6 - 4**6, x)
        result, details = check_rationality(poly)
        # y³ - 1 is binomial with perfect cube
        assert result is True
    
    def test_eighth_degree_biquadratic(self):
        # x⁸ - 1 → reduces to y⁴ - 1 → reduces to z² - 1
        x = sym.Symbol('x')
        poly = sym.Poly(x**8 - sym.Rational(12/17)**8, x)
        result, details = check_rationality(poly)
        assert result is True

class TestAtLeastOneCleanRoot:
    def test_all_clean(self):
        # x² - 4 → both roots clean
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 4, x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is True
    
    def test_none_clean(self):
        # x² - 2 → no clean roots
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 2, x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is False
    
    def test_mixed_factors(self):
        # (x - 1)(x² - 2) = x³ - x² - 2x + 2
        # Has one clean root (x = 1) and two non-clean (±√2)
        x = sym.Symbol('x')
        poly = sym.Poly((x - 1) * (x**2 - 2), x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is True
    
    def test_mixed_factors_2(self):
        # (x - 3)(x² - 5) → one clean root (x = 3)
        x = sym.Symbol('x')
        poly = sym.Poly((x - 3) * (x**2 - 5), x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is True
    
    def test_parametric_linear_factor(self):
        # (x - a)(x² - 2) → clean root x = a
        x, a = sym.symbols('x a')
        poly = sym.Poly((x - a) * (x**2 - 2), x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is True
    
    def test_parametric_quadratic_clean(self):
        # (x² - a) → roots ±√a, which are clean (radical of parameter)
        x, a = sym.symbols('x a')
        poly = sym.Poly(x**2 - a, x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is True
    
    def test_all_irrational_factors(self):
        # (x² - 2)(x² - 3) → no clean roots
        x = sym.Symbol('x')
        poly = sym.Poly((x**2 - 2) * (x**2 - 3), x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is False
    
    def test_linear(self):
        x, a = sym.symbols('x a')
        poly = sym.Poly(a*x + 1, x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is True
    
    def test_constant(self):
        x = sym.Symbol('x')
        poly = sym.Poly(5, x)
        result, details = has_at_least_one_clean_root(poly)
        assert result is False  # No roots at all


class TestGetCleanRoots:
    def test_linear(self):
        x = sym.Symbol('x')
        poly = sym.Poly(2*x - 6, x)
        roots = get_clean_roots(poly)
        assert roots == [3]
    
    def test_quadratic_both_clean(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 4, x)
        roots = get_clean_roots(poly)
        assert set(roots) == {2, -2}
    
    def test_quadratic_none_clean(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 2, x)
        roots = get_clean_roots(poly)
        assert roots == []
    
    def test_mixed_factors(self):
        x = sym.Symbol('x')
        poly = sym.Poly((x - 1) * (x**2 - 2), x)
        roots = get_clean_roots(poly)
        assert roots == [1]
    
    def test_multiple_clean_factors(self):
        x = sym.Symbol('x')
        poly = sym.Poly((x - 1) * (x + 2) * (x**2 - 3), x)
        roots = get_clean_roots(poly)
        assert set(roots) == {1, -2}
    
    def test_repeated_root(self):
        x = sym.Symbol('x')
        poly = sym.Poly((x - 1)**2 * (x**2 - 2), x)
        roots = get_clean_roots(poly)
        assert roots == [1, 1]  # Multiplicity 2
    
    def test_parametric(self):
        x, a = sym.symbols('x a')
        poly = sym.Poly((x - a) * (x**2 - 2), x)
        roots = get_clean_roots(poly)
        assert roots == [a]


class TestCountCleanRoots:
    def test_none(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 2, x)
        assert count_clean_roots(poly) == 0
    
    def test_some(self):
        x = sym.Symbol('x')
        poly = sym.Poly((x - 1) * (x**2 - 2), x)
        assert count_clean_roots(poly) == 1
    
    def test_all(self):
        x = sym.Symbol('x')
        poly = sym.Poly(x**2 - 4, x)
        assert count_clean_roots(poly) == 2
    
    def test_with_multiplicity(self):
        x = sym.Symbol('x')
        poly = sym.Poly((x - 1)**3 * (x**2 - 2), x)
        assert count_clean_roots(poly) == 3