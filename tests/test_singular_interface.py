"""
Tests for singular_interface module.

Tests the Groebner Cover functionality using examples from the
Singular grobcov.lib documentation.

Author: Eli Weissler
"""

import pytest
from sircuitenum.singular_interface import (
    groebner_cover,
    format_grobcov_input,
    parse_grobcov_output,
    run_grobcov,
    summarize_segments
)

CASES_ALVERO_DOCS = """
                    ==> [1]:
                    ==>    [1]:
                    ==>       _[1]=1
                    ==>    [2]:
                    ==>       _[1]=1
                    ==>    [3]:
                    ==>       [1]:
                    ==>          [1]:
                    ==>             _[1]=0
                    ==>          [2]:
                    ==>             [1]:
                    ==>                _[1]=(a2-a3^2)
                    ==>                _[2]=(a1-a3^3)
                    ==>                _[3]=(a0-a3^4)
                    ==> [2]:
                    ==>    [1]:
                    ==>       _[1]=x3
                    ==>       _[2]=x2^2
                    ==>       _[3]=x1^3
                    ==>    [2]:
                    ==>       _[1]=x3+(a3)
                    ==>       _[2]=x2^2+(2*a3)*x2+(a3^2)
                    ==>       _[3]=x1^3+(3*a3)*x1^2+(3*a3^2)*x1+(a3^3)
                    ==>    [3]:
                    ==>       [1]:
                    ==>          [1]:
                    ==>             _[1]=(a2-a3^2)
                    ==>             _[2]=(a1-a3^3)
                    ==>             _[3]=(a0-a3^4)
                    ==>          [2]:
                    ==>             [1]:
                    ==>                _[1]=1
                    """

ROBOT_DOCS = """
        ==> [1]:
        ==>    [1]:
        ==>       _[1]=c1
        ==>       _[2]=s3
        ==>       _[3]=c3
        ==>       _[4]=s1^2
        ==>    [2]:
        ==>       _[1]=(2*a*l2)*c1+(2*b*l2)*s1+(-a^2-b^2-l2^2+l3^2)
        ==>       _[2]=(l3)*s3+(l2)*s1+(-b)
        ==>       _[3]=(2*a*l3)*c3+(-2*b*l2)*s1+(-a^2+b^2+l2^2-l3^2)
        ==>       _[4]=(4*a^2*l2^2+4*b^2*l2^2)*s1^2+(-4*a^2*b*l2-4*b^3*l2-4*b*l2^3+4*b*l2*l3^2)*s1+(a^4+2*a^2*b^2-2*a^2*l2^2-2*a^2*l3^2+b^4+2*b^2*l2^2-2*b^2*l3^2+l2^4-2*l2^2*l3^2+l3^4)
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=0
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>             [2]:
        ==>                _[1]=(l2)
        ==>             [3]:
        ==>                _[1]=(a^2+b^2)
        ==>             [4]:
        ==>                _[1]=(a)
        ==> [2]:
        ==>    [1]:
        ==>       _[1]=s1
        ==>       _[2]=s3
        ==>       _[3]=c3
        ==>       _[4]=c1^2
        ==>    [2]:
        ==>       _[1]=(2*b*l2)*s1+(-b^2-l2^2+l3^2)
        ==>       _[2]=(2*b*l3)*s3+(-b^2+l2^2-l3^2)
        ==>       _[3]=(l3)*c3+(l2)*c1
        ==>       _[4]=(4*b^2*l2^2)*c1^2+(b^4-2*b^2*l2^2-2*b^2*l3^2+l2^4-2*l2^2*l3^2+l3^4)
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(a)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(a)
        ==>             [2]:
        ==>                _[1]=(l2)
        ==>                _[2]=(a)
        ==>             [3]:
        ==>                _[1]=(b)
        ==>                _[2]=(a)
        ==> [3]:
        ==>    [1]:
        ==>       _[1]=1
        ==>    [2]:
        ==>       _[1]=1
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(b)
        ==>             _[2]=(a)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l2+l3)
        ==>                _[2]=(b)
        ==>                _[3]=(a)
        ==>             [2]:
        ==>                _[1]=(l3)
        ==>                _[2]=(b)
        ==>                _[3]=(a)
        ==>             [3]:
        ==>                _[1]=(l2-l3)
        ==>                _[2]=(b)
        ==>                _[3]=(a)
        ==>       [2]:
        ==>          [1]:
        ==>             _[1]=(l2)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l2)
        ==>                _[2]=(a^2+b^2-l3^2)
        ==>             [2]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==> [4]:
        ==>    [1]:
        ==>       _[1]=s3
        ==>       _[2]=c3
        ==>       _[3]=c1^2
        ==>    [2]:
        ==>       _[1]=(l2^2*l3+2*l2^2-l3^3)*s3+(2*l2*l3)*s1+(b*l3^2)
        ==>       _[2]=(l2^2*l3+2*l2^2-l3^3)*c3+(2*l2*l3)*c1+(a*l3^2)
        ==>       _[3]=c1^2+s1^2-1
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(l2-l3)
        ==>             _[2]=(b)
        ==>             _[3]=(a)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==>                _[3]=(b)
        ==>                _[4]=(a)
        ==>       [2]:
        ==>          [1]:
        ==>             _[1]=(l2+l3)
        ==>             _[2]=(b)
        ==>             _[3]=(a)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==>                _[3]=(b)
        ==>                _[4]=(a)
        ==>       [3]:
        ==>          [1]:
        ==>             _[1]=(l2)
        ==>             _[2]=(a^2+b^2-l3^2)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==>                _[3]=(a^2+b^2)
        ==> [5]:
        ==>    [1]:
        ==>       _[1]=c1^2
        ==>       _[2]=c3^2
        ==>    [2]:
        ==>       _[1]=c1^2+s1^2-1
        ==>       _[2]=c3^2+s3^2-1
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(l3)
        ==>             _[2]=(l2)
        ==>             _[3]=(b)
        ==>             _[4]=(a)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=1
        ==> [6]:
        ==>    [1]:
        ==>       _[1]=1
        ==>    [2]:
        ==>       _[1]=1
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(l3)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(a^2+b^2-l2^2)
        ==>             [2]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==> [7]:
        ==>    [1]:
        ==>       _[1]=1
        ==>    [2]:
        ==>       _[1]=1
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(l3)
        ==>             _[2]=(l2)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==>                _[3]=(a^2+b^2)
        ==> [8]:
        ==>    [1]:
        ==>       _[1]=s1
        ==>       _[2]=c1
        ==>       _[3]=c3^2
        ==>    [2]:
        ==>       _[1]=(l2)*s1+(-b)
        ==>       _[2]=(l2)*c1+(-a)
        ==>       _[3]=c3^2+s3^2-1
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(l3)
        ==>             _[2]=(a^2+b^2-l2^2)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==>                _[3]=(a^2+b^2)
        ==> [9]:
        ==>    [1]:
        ==>       _[1]=1
        ==>    [2]:
        ==>       _[1]=1
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(l3)
        ==>             _[2]=(l2)
        ==>             _[3]=(a^2+b^2)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==>                _[3]=(b)
        ==>                _[4]=(a)
        ==> [10]:
        ==>    [1]:
        ==>       _[1]=s1
        ==>       _[2]=c1
        ==>       _[3]=s3
        ==>       _[4]=c3
        ==>    [2]:
        ==>       _[1]=(4*b*l2^3-4*b*l2*l3^2)*s1+(-4*b^2*l2^2-l2^4+2*l2^2*l3^2-l3^4)
        ==>       _[2]=(4*b^2*l2^3-4*b^2*l2*l3^2)*c1+(-4*a*b^2*l2^2+a*l2^4-2*a*l2^2*l3^2+a*l3^4)
        ==>       _[3]=(4*b*l2^2*l3-4*b*l3^3)*s3+(4*b^2*l3^2+l2^4-2*l2^2*l3^2+l3^4)
        ==>       _[4]=(4*b^2*l2^2*l3-4*b^2*l3^3)*c3+(4*a*b^2*l3^2-a*l2^4+2*a*l2^2*l3^2-a*l3^4)
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(a^2+b^2)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l2+l3)
        ==>                _[2]=(a^2+b^2)
        ==>             [2]:
        ==>                _[1]=(l3)
        ==>                _[2]=(a^2+b^2)
        ==>             [3]:
        ==>                _[1]=(l2-l3)
        ==>                _[2]=(a^2+b^2)
        ==>             [4]:
        ==>                _[1]=(l2)
        ==>                _[2]=(a^2+b^2)
        ==>             [5]:
        ==>                _[1]=(b)
        ==>                _[2]=(a)
        ==> [11]:
        ==>    [1]:
        ==>       _[1]=1
        ==>    [2]:
        ==>       _[1]=1
        ==>    [3]:
        ==>       [1]:
        ==>          [1]:
        ==>             _[1]=(l2-l3)
        ==>             _[2]=(a^2+b^2)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==>                _[3]=(a^2+b^2)
        ==>             [2]:
        ==>                _[1]=(l2-l3)
        ==>                _[2]=(b)
        ==>                _[3]=(a)
        ==>       [2]:
        ==>          [1]:
        ==>             _[1]=(l2+l3)
        ==>             _[2]=(a^2+b^2)
        ==>          [2]:
        ==>             [1]:
        ==>                _[1]=(l3)
        ==>                _[2]=(l2)
        ==>                _[3]=(a^2+b^2)
        ==>             [2]:
        ==>                _[1]=(l2+l3)
        ==>                _[2]=(b)
        ==>                _[3]=(a)
        """

# Grob1Levels Documentation Example
# https://www.singular.uni-kl.de/Manual/4-4/sing_1073.htm#SEC1154
GROB1LEVELS_DOCS = """
        ==> [1]:
        ==>    [1]:
        ==>       _[1]=1
        ==>    [2]:
        ==>       _[1]=1
        ==>    [3]:
        ==>       [1]:
        ==>          _[1]=0
        ==>       [2]:
        ==>          _[1]=(x^5*y-2*x^3*y-x*y^5+x*y)
        ==> [2]:
        ==>    [1]:
        ==>       _[1]=y2
        ==>       _[2]=x2
        ==>       _[3]=y1
        ==>       _[4]=x1
        ==>    [2]:
        ==>       _[1]=(x^5+2*x^4*y^6+4*x^4*y^4+2*x^4*y^2+4*x^3*y^6+8*x^3*y^4+4*x^3*y\
        ^2-2*x^3-4*x^2*y^6-8*x^2*y^4-4*x^2*y^2-4*x*y^6-9*x*y^4-4*x*y^2+x-2*y^10-4\
        *y^8+4*y^4+2*y^2)*y2+(-6*x^4*y^5-6*x^4*y^3-2*x^3*y^7-4*x^3*y^5-2*x^3*y^3-\
        2*x^2*y^7+8*x^2*y^5+10*x^2*y^3+2*x*y^9+6*x*y^7+6*x*y^5+2*x*y^3+4*y^9+4*y^\
        7-4*y^5-4*y^3)
        ==>       _[2]=(x^5+2*x^4*y^7+4*x^4*y^5+2*x^4*y^3+4*x^3*y^7+8*x^3*y^5+4*x^3*y\
        ^3-2*x^3-4*x^2*y^7-8*x^2*y^5-4*x^2*y^3-4*x*y^7-8*x*y^5-x*y^4-4*x*y^3+x-2*\
        y^11-4*y^9+4*y^5+2*y^3)*x2+(-x^5-6*x^4*y^5-6*x^4*y^3+2*x^3+2*x^2*y^9+6*x^\
        2*y^5+8*x^2*y^3+x*y^4-x-2*y^11+4*y^7-2*y^3)
        ==>       _[3]=(x^5+2*x^4*y^6+4*x^4*y^4+2*x^4*y^2+4*x^3*y^6+8*x^3*y^4+4*x^3*y\
        ^2-2*x^3-4*x^2*y^6-8*x^2*y^4-4*x^2*y^2-4*x*y^6-9*x*y^4-4*x*y^2+x-2*y^10-4\
        *y^8+4*y^4+2*y^2)*y1+(-2*x^4*y^5-2*x^4*y^3-2*x^3*y^7-4*x^3*y^5-2*x^3*y^3+\
        2*x^2*y^7+8*x^2*y^5+6*x^2*y^3+2*x*y^9+6*x*y^7+6*x*y^5+2*x*y^3+4*y^9+4*y^7\
        -4*y^5-4*y^3)
        ==>       _[4]=(x^5+2*x^4*y^7+4*x^4*y^5+2*x^4*y^3+4*x^3*y^7+8*x^3*y^5+4*x^3*y\
        ^3-2*x^3-4*x^2*y^7-8*x^2*y^5-4*x^2*y^3-4*x*y^7-8*x*y^5-x*y^4-4*x*y^3+x-2*\
        y^11-4*y^9+4*y^5+2*y^3)*x1+(x^5-4*x^4*y^7-6*x^4*y^5-2*x^4*y^3-2*x^3+2*x^2\
        *y^9+8*x^2*y^7+6*x^2*y^5-x*y^4+x+2*y^11-4*y^7+2*y^3)
        ==>    [3]:
        ==>       [1]:
        ==>          _[1]=(x^5*y-2*x^3*y-x*y^5+x*y)
        ==>       [2]:
        ==>          _[1]=(x*y)
        ==>          _[2]=(x^2-y^2-1)
        ==>          _[3]=(y^3+y)
        ==> [3]:
        ==>    [1]:
        ==>       _[1]=y2^2
        ==>       _[2]=y1
        ==>       _[3]=x1
        ==>    [2]:
        ==>       _[1]=y2^2
        ==>       _[2]=y1
        ==>       _[3]=x1+1
        ==>    [3]:
        ==>       [1]:
        ==>          _[1]=(y)
        ==>          _[2]=(x+1)
        ==>       [2]:
        ==>          _[1]=1
        ==> [4]:
        ==>    [1]:
        ==>       _[1]=y2
        ==>       _[2]=x2
        ==>       _[3]=y1^2
        ==>    [2]:
        ==>       _[1]=y2
        ==>       _[2]=x2-1
        ==>       _[3]=y1^2
        ==>    [3]:
        ==>       [1]:
        ==>          _[1]=(y)
        ==>          _[2]=(x-1)
        ==>       [2]:
        ==>          _[1]=1
        ==> [5]:
        ==>    [1]:
        ==>       _[1]=1
        ==>    [2]:
        ==>       _[1]=1
        ==>    [3]:
        ==>       [1]:
        ==>          _[1]=(x)
        ==>          _[2]=(y^2+1)
        ==>       [2]:
        ==>          _[1]=1
        """

class TestFormatGrobcovInput:
    """Tests for format_grobcov_input function."""

    def test_simple_single_variable(self):
        """Test formatting with single solve variable and two parameters."""
        cmd = format_grobcov_input(
            ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
            solve_vars=["Z01"],
            param_vars=["Z10", "Z11"]
        )
        assert 'LIB "grobcov.lib"' in cmd
        assert "ring R = (0,Z10,Z11),(Z01),dp" in cmd
        assert "ideal F = Z01*(Z10+1) + Z10*Z11" in cmd
        assert 'grobcov(F,"rep",2,"ext",1,"comment",0)' in cmd

    def test_multiple_solve_variables(self):
        """Test formatting with multiple solve variables."""
        cmd = format_grobcov_input(
            ideal_gens=["x1^2 + x2^2 - 1", "x1 - a"],
            solve_vars=["x1", "x2"],
            param_vars=["a"]
        )
        assert "(0,a),(x1,x2)" in cmd
        assert "x1^2 + x2^2 - 1, x1 - a" in cmd

    def test_custom_ordering(self):
        """Test with lexicographic ordering."""
        cmd = format_grobcov_input(
            ideal_gens=["x + y"],
            solve_vars=["x", "y"],
            param_vars=["a"],
            ordering="lp"
        )
        assert "(x,y),lp" in cmd

    def test_rep_options(self):
        """Test different representation options."""
        cmd_p = format_grobcov_input(
            ideal_gens=["x"],
            solve_vars=["x"],
            param_vars=["a"],
            rep=0
        )
        assert '"rep",0' in cmd_p

        cmd_c = format_grobcov_input(
            ideal_gens=["x"],
            solve_vars=["x"],
            param_vars=["a"],
            rep=1
        )
        assert '"rep",1' in cmd_c


class TestParseGroebnerCover:
    """Tests for parse_grobcov_output function."""

    def test_single_segment_simple(self):
        """Test parsing a simple single-segment output with C-representation."""
        # C-representation format: [4]:[1]: for E, [4]:[2]: for N
        text = """[1]:
   [1]:
      _[1]=Z01
   [2]:
      _[1]=(Z10+1)*Z01+(Z10*Z11)
   [4]:
      [1]:
         _[1]=0
      [2]:
         _[1]=(Z10+1)"""
        
        segments = parse_grobcov_output(text)
        assert len(segments) == 1
        assert segments[0]['lpp'] == ['Z01']
        assert segments[0]['basis'] == ['(Z10+1)*Z01+(Z10*Z11)']
        assert segments[0]['N'] == ['Z10+1']
        assert segments[0]['E'] == []

    def test_multiple_segments(self):
        """Test parsing output with multiple segments."""
        text = """[1]:
   [1]:
      _[1]=Z01
   [2]:
      _[1]=(Z10+1)*Z01+(Z10*Z11)
   [3]:
      [1]:
         [1]:
            _[1]=0
         [2]:
            [1]:
               _[1]=(Z10+1)
[2]:
   [1]:
      _[1]=1
   [2]:
      _[1]=1
   [3]:
      [1]:
         [1]:
            _[1]=(Z10+1)
            _[2]=(Z11)
         [2]:
            [1]:
               _[1]=1"""
        
        segments = parse_grobcov_output(text)
        assert len(segments) == 2
        
        # First segment: solvable
        assert segments[0]['lpp'] == ['Z01']
        
        # Second segment: no solutions (basis = [1])
        assert segments[1]['lpp'] == ['1']
        assert segments[1]['basis'] == ['1']
        assert 'Z10+1' in segments[1]['E']
        assert 'Z11' in segments[1]['E']


    def test_c_representation_preferred(self):
        """Test that C-representation is preferred over P-representation."""
        # C-rep ([4]:) is flat: [1]: for E, [2]: for N
        # P-rep ([3]:) is nested: [1]:[1]: for E, [1]:[2]: for N
        text = """[1]:
   [1]:
      _[1]=x
   [2]:
      _[1]=(a)*x+(b)
   [3]:
      [1]:
         [1]:
            _[1]=0
         [2]:
            [1]:
               _[1]=(a)
   [4]:
      [1]:
         _[1]=0
      [2]:
         _[1]=(a)"""
        
        segments = parse_grobcov_output(text)
        assert len(segments) == 1
        # Should use C-rep (from [4]:) which is same as P-rep here
        assert segments[0]['N'] == ['a']

    def test_alvero_example(self):
        """Test parsing the Casas-Alvero example output."""
        text = CASES_ALVERO_DOCS.replace("                    ==> ", "")
        segments = parse_grobcov_output(text)
        assert len(segments) == 2
        # Check first segment
        assert segments[0]['lpp'] == ['1']
        # Check second segment
        assert segments[1]['lpp'] == ['x3', 'x2^2', 'x1^3']

    def test_robot_example(self):
        text = ROBOT_DOCS.replace("        ==> ", "")
        segments = parse_grobcov_output(text)
        assert len(segments) == 11
        # Check first segment
        assert segments[0]['lpp'] == ['c1', 's3', 'c3', 's1^2']
        assert segments[0]['basis'] == ["(2*a*l2)*c1+(2*b*l2)*s1+(-a^2-b^2-l2^2+l3^2)",
                                        "(l3)*s3+(l2)*s1+(-b)",
                                        "(2*a*l3)*c3+(-2*b*l2)*s1+(-a^2+b^2+l2^2-l3^2)",
                                        "(4*a^2*l2^2+4*b^2*l2^2)*s1^2+(-4*a^2*b*l2-4*b^3*l2-4*b*l2^3+4*\
   b*l2*l3^2)*s1+(a^4+2*a^2*b^2-2*a^2*l2^2-2*a^2*l3^2+b^4+2*b^2*l2^2-2*b^2*l\
   3^2+l2^4-2*l2^2*l3^2+l3^4)"]
        assert segments[0]['N'] == []
        assert segments[0]['E'] == ['0', 'l3', 'l2', 'a^2+b^2', 'a']
        # Check second segment
        assert segments[1]['lpp'] == ['s1', 's3', 'c3', 'c1^2']
        assert segments[1]['basis'] == ["(2*b*l2)*s1+(-b^2-l2^2+l3^2)",
                                        "(2*b*l3)*s3+(-b^2+l2^2-l3^2)",
                                        "(l3)*c3+(l2)*c1",
                                        "(4*b^2*l2^2)*c1^2+(b^4-2*b^2*l2^2-2*b^2*l3^2+l2^4-2*l2^2*l3^2+l3^4)"]
        assert segments[1]['N'] == []
        assert segments[1]['E'] == ['a', 'l3', 'l2', 'b']
        # Check third segment
        assert segments[2]['lpp'] == ['1']
        # Check fourth segment
        ## TODO: Figure out how to properly parse nested E and N
        assert segments[3]['lpp'] == ['s3', 'c3', 'c1^2']
        assert segments[3]['basis'] == ["(l2^2*l3+2*l2^2-l3^3)*s3+(2*l2*l3)*s1+(b*l3^2)",
                                        "(l2^2*l3+2*l2^2-l3^3)*c3+(2*l2*l3)*c1+(a*l3^2)",
                                        "c1^2+s1^2-1"]
        assert segments[3]['N'] == []
        assert segments[3]['E'] == ['l2-l3', 'b', 'a', 'l3', 'l2', 'a^2+b^2']
        # Check fifth segment
        assert segments[4]['lpp'] == ['c1^2', 'c3^2']
        # Check sixth segment
        assert segments[5]['lpp'] == ['1']
        # Check seventh segment
        assert segments[6]['lpp'] == ['1']
        # Check eighth segment
        assert segments[7]['lpp'] == ['s1', 'c1', 'c3^2']
        # Check ninth segment
        assert segments[8]['lpp'] == ['1']
        # Check tenth segment
        assert segments[9]['lpp'] == ['s1', 'c1', 's3', 'c3']

    def test_grob1levels_example(self):
        """Test parsing the Grob1Levels documentation example."""
        text = GROB1LEVELS_DOCS.replace("        ==> ", "")
        segments = parse_grobcov_output(text)
        assert len(segments) == 5
        # Check first segment
        assert segments[0]['lpp'] == ['1']
        # Check second segment
        assert segments[1]['lpp'] == ['y2', 'x2', 'y1', 'x1']
        # Check third segment
        assert segments[2]['lpp'] == ['y2^2', 'y1', 'x1']
        # Check fourth segment
        assert segments[3]['lpp'] == ['y2', 'x2', 'y1^2']
        # Check fifth segment
        assert segments[4]['lpp'] == ['1']


class TestRunGrobcov:
    """Tests for run_grobcov function."""

    def test_returns_string_with_segments(self):
        """Test that run_grobcov returns a string containing segment markers."""
        output = run_grobcov(
            ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
            solve_vars=["Z01"],
            param_vars=["Z10", "Z11"]
        )
        # Should return a string
        assert isinstance(output, str)
        # Should contain segment marker as per docstring example
        assert "[1]:" in output

    def test_output_contains_basis(self):
        """Test that output contains basis polynomials."""
        output = run_grobcov(
            ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
            solve_vars=["Z01"],
            param_vars=["Z10", "Z11"]
        )
        # Should contain the basis polynomial
        assert "Z01" in output
        assert "Z10" in output

    def test_output_parseable(self):
        """Test that run_grobcov output can be parsed by parse_grobcov_output."""
        output = run_grobcov(
            ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
            solve_vars=["Z01"],
            param_vars=["Z10", "Z11"]
        )
        # Should be parseable
        segments = parse_grobcov_output(output)
        assert len(segments) == 3
        assert segments[0]['lpp'] == ['Z01']

    def test_multiple_segments_in_output(self):
        """Test that output contains multiple segment markers for multi-segment problems."""
        output = run_grobcov(
            ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
            solve_vars=["Z01"],
            param_vars=["Z10", "Z11"]
        )
        # Should have 3 segments
        assert "[1]:" in output
        assert "[2]:" in output
        assert "[3]:" in output

    def test_casas_alvero_example(self):
        """Test groebner_cover on Casas-Alvero conjecture example."""
        output = run_grobcov(
            ideal_gens=[
                "x1^4+(4*a3)*x1^3+(6*a2)*x1^2+(4*a1)*x1+(a0)",
                "x1^3+(3*a3)*x1^2+(3*a2)*x1+(a1)",
                "x2^4+(4*a3)*x2^3+(6*a2)*x2^2+(4*a1)*x2+(a0)",
                "x2^2+(2*a3)*x2+(a2)",
                "x3^4+(4*a3)*x3^3+(6*a2)*x3^2+(4*a1)*x3+(a0)",
                "x3+(a3)"
            ],
            solve_vars=["x1", "x2", "x3"],
            param_vars=["a0", "a1", "a2", "a3"],
            ordering="dp",
            verbose=True,
            rep=0,
            ext=0
        )

        from_docs = CASES_ALVERO_DOCS    

        # Normalize output for comparison
        norm_output = "\n".join(line.strip("=> ") for line in output.splitlines() if line.strip() and line[0] != "/")
        norm_docs = "\n".join(line.strip("=> ") for line in from_docs.splitlines() if line.strip())
    
        assert norm_docs == norm_output

    def test_robot_example(self):

        ideal_gens = [
            "a-l3*c3-l2*c1",      # x-coordinate constraint
            "b-l3*s3-l2*s1",      # y-coordinate constraint  
            "c1^2+s1^2-1",        # unit circle for joint 1
            "c3^2+s3^2-1"         # unit circle for joint 3
        ]
        
        output = run_grobcov(
            ideal_gens=ideal_gens,
            solve_vars=["c3", "s3", "c1", "s1"],
            param_vars=["a", "b", "l2", "l3"],
            ordering="dp",
            rep=0,
            ext=0
        )

        from_docs = ROBOT_DOCS

        # Normalize output for comparison
        norm_output = "\n".join(line.strip("=> ") for line in output.splitlines() if line.strip() and line[0] != "/")
        norm_docs = "\n".join(line.strip("=> ") for line in from_docs.splitlines() if line.strip())
        assert norm_docs == norm_output

class TestGroebnerCover:
    """Integration tests for groebner_cover function."""

    # TODO: Make these tests more specific by checking segment contents
    def test_casas_alvero_degree4(self):
        """
        EXAMPLE 1 from grobcov.lib: Casas-Alvero conjecture for degree 4.
        
        Casas-Alvero conjecture states that on a field of characteristic 0,
        if a polynomial of degree n in x has a common root with each of its
        n-1 derivatives (not assumed to be the same), then it is of the form
        P(x) = k(x + a)^n.
        """
        # Polynomial: x^4 + 4*a3*x^3 + 6*a2*x^2 + 4*a1*x + a0
        # Shares root with 1st derivative at x1
        # Shares root with 2nd derivative at x2  
        # Shares root with 3rd derivative at x3
        ideal_gens = [
            "x1^4+(4*a3)*x1^3+(6*a2)*x1^2+(4*a1)*x1+(a0)",
            "x1^3+(3*a3)*x1^2+(3*a2)*x1+(a1)",
            "x2^4+(4*a3)*x2^3+(6*a2)*x2^2+(4*a1)*x2+(a0)",
            "x2^2+(2*a3)*x2+(a2)",
            "x3^4+(4*a3)*x3^3+(6*a2)*x3^2+(4*a1)*x3+(a0)",
            "x3+(a3)"
        ]
        
        segments = groebner_cover(
            ideal_gens=ideal_gens,
            solve_vars=["x1", "x2", "x3"],
            param_vars=["a0", "a1", "a2", "a3"],
            ordering="dp"
        )
        
        # Should have 2 segments according to the documentation
        assert len(segments) == 2
        
        # Segment 1: No solutions (basis = [1])
        # This is the "generic" case where the conjecture fails
        assert segments[0]['lpp'] == ['1']
        assert segments[0]['basis'] == ['1']
        
        # Segment 2: The conjecture holds when a0 = a3^4, a1 = a3^3, a2 = a3^2
        # i.e., the polynomial is (x + a3)^4
        assert 'x3' in segments[1]['lpp']
        
        # Check the guards contain the expected equalities
        guards_e = segments[1]['E']
        # Should have conditions like a2-a3^2, a1-a3^3, a0-a3^4
        guard_str = " ".join(guards_e)
        assert 'a2' in guard_str or len(guards_e) > 0

    def test_robot_arm(self):
        """
        EXAMPLE 2 from grobcov.lib: Two-arm robot problem (M. Rychlik).
        
        Robot arm with two links of lengths l2 and l3, trying to reach point (a, b).
        Variables: c1, s1 (cos/sin of joint 1), c3, s3 (cos/sin of joint 3).
        """
        ideal_gens = [
            "a-l3*c3-l2*c1",      # x-coordinate constraint
            "b-l3*s3-l2*s1",      # y-coordinate constraint  
            "c1^2+s1^2-1",        # unit circle for joint 1
            "c3^2+s3^2-1"         # unit circle for joint 3
        ]
        
        segments = groebner_cover(
            ideal_gens=ideal_gens,
            solve_vars=["c3", "s3", "c1", "s1"],
            param_vars=["a", "b", "l2", "l3"],
            ordering="dp"
        )
        
        # Should have 11 segments according to the documentation
        assert len(segments) == 11
        
        # Check some key segments exist:
        # - Segments with basis [1] (no solutions)
        # - Segments with actual solutions
        no_solution_count = sum(1 for s in segments if s['basis'] == ['1'])
        assert no_solution_count > 0
        
        # Check that some segments have non-trivial bases
        has_nontrivial = any(
            s['basis'] != ['1'] and len(s['basis']) > 0 
            for s in segments
        )
        assert has_nontrivial

    def test_simple_linear_equation(self):
        """Test simple linear equation Z01*(Z10+1) + Z10*Z11 = 0."""
        segments = groebner_cover(
            ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
            solve_vars=["Z01"],
            param_vars=["Z10", "Z11"]
        )
        
        # Should have 3 segments:
        # 1. Z10+1 ≠ 0: Z01 = -Z10*Z11/(Z10+1)
        # 2. Z10+1 = 0, Z11 ≠ 0: no solutions
        # 3. Z10+1 = 0, Z11 = 0: Z01 free
        assert len(segments) == 3
        
        # Find the "main" segment (Z10+1 ≠ 0 in N, no E)
        main_segment = None
        for s in segments:
            if 'Z10+1' in s['N'] and not s['E']:
                main_segment = s
                break
        
        assert main_segment is not None
        assert main_segment['lpp'] == ['Z01']
        
        # Find the "no solution" segment (basis = [1])
        no_sol_segment = None
        for s in segments:
            if s['basis'] == ['1']:
                no_sol_segment = s
                break
        
        assert no_sol_segment is not None
        assert 'Z10+1' in no_sol_segment['E']
        # Z11 should be in N (inequality) for no-solution segment
        assert 'Z11' in no_sol_segment['N'] or 'Z11' in no_sol_segment['E']

    def test_circle_line_intersection(self):
        """Test intersection of circle and line with parameter."""
        # Circle: x^2 + y^2 - 1 = 0
        # Line: y - a = 0 (horizontal line at height a)
        segments = groebner_cover(
            ideal_gens=["x^2 + y^2 - 1", "y - a"],
            solve_vars=["x", "y"],
            param_vars=["a"]
        )
        
        # Groebner cover produces a single segment with basis:
        # y = a, x^2 = 1 - a^2
        # The number of solutions depends on sign of (1 - a^2), but
        # grobcov doesn't split on discriminants unless the leading
        # power products change.
        assert len(segments) >= 1
        
        # The segment should have the triangular basis
        assert any(
            'y' in s['lpp'] and 'x2' in s['lpp']
            for s in segments
        )


class TestSummarizeSegments:
    """Tests for summarize_segments function."""

    def test_summarize_prints_output(self, capsys):
        """Test that summarize_segments prints readable output."""
        segments = [
            {
                'lpp': ['Z01'],
                'basis': ['(Z10+1)*Z01+(Z10*Z11)'],
                'E': [],
                'N': ['Z10+1'],
                'E_p': [], 'N_p': ['Z10+1'],
                'E_c': [], 'N_c': ['Z10+1']
            },
            {
                'lpp': ['1'],
                'basis': ['1'],
                'E': ['Z10+1', 'Z11'],
                'N': [],
                'E_p': ['Z10+1', 'Z11'], 'N_p': [],
                'E_c': ['Z10+1', 'Z11'], 'N_c': []
            }
        ]
        
        summarize_segments(segments)
        captured = capsys.readouterr()
        
        assert "Segment 1:" in captured.out
        assert "Segment 2:" in captured.out
        assert "lpp:" in captured.out
        assert "basis:" in captured.out
        assert "guards:" in captured.out
        assert "No solutions" in captured.out


class TestEdgeCases:
    """Edge case tests."""

    def test_empty_ideal(self):
        """Test with trivial ideal."""
        segments = groebner_cover(
            ideal_gens=["0"],
            solve_vars=["x"],
            param_vars=["a"]
        )
        # Should return at least one segment
        assert len(segments) >= 1

    def test_constant_ideal(self):
        """Test with constant (no parameter dependence)."""
        segments = groebner_cover(
            ideal_gens=["x^2 - 1"],
            solve_vars=["x"],
            param_vars=["a"]  # a doesn't appear
        )
        # Should give same result regardless of a
        assert len(segments) >= 1


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
