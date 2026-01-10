"""
Singular Interface for Groebner Cover computation.

This module provides a Python/Sage interface to the Singular grobcov library
for computing the Groebner Cover of a parametric polynomial ideal.

Author: Eli Weissler
Version: 0.1.0
"""

__all__ = ["groebner_cover", "run_grobcov", "parse_grobcov_output", "format_grobcov_input"]

import re
from typing import List, Dict, Any, Sequence, Union

from sage.interfaces.singular import singular


def format_grobcov_input(
    ideal_gens: Sequence[str],
    solve_vars: Sequence[str],
    param_vars: Sequence[str],
    ordering: str = "dp",
    rep: int = 2,
    ext: int = 1,
    comment: int = 0,
) -> str:
    """
    Format the Singular command string to run grobcov on a parametric ideal.

    Parameters
    ----------
    ideal_gens : Sequence[str]
        List of polynomial generators as strings (in Singular syntax).
    solve_vars : Sequence[str]
        Variables to solve for (appear in the polynomial ring).
    param_vars : Sequence[str]
        Parameter variables (appear in the coefficient field).
    ordering : str, optional
        Monomial ordering for the polynomial ring. Default is "dp".
    rep : int, optional
        Representation type for grobcov output:
        - 0: P-representation only
        - 1: C-representation only  
        - 2: Both P and C representations (default)
    ext : int, optional
        Extension level for the output. Default is 1.
    comment : int, optional
        Level of verbosity for grobcov progress. Default is 0 (no comments).
        Goes up to 3 for more detailed output.

    Returns
    -------
    str
        The Singular command string to execute.

    Examples
    --------
    >>> cmd = format_grobcov_input(
    ...     ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
    ...     solve_vars=["Z01"],
    ...     param_vars=["Z10", "Z11"]
    ... )
    >>> print(cmd)
    LIB "grobcov.lib"; ring R = (0,Z10,Z11),(Z01),dp; ideal F = Z01*(Z10+1) + Z10*Z11; list C = grobcov(F,"rep",2,"ext",1); C;
    """
    params_str = ",".join(param_vars)
    vars_str = ",".join(solve_vars)
    gens_str = ", ".join(ideal_gens)
    
    cmd = (
        f'LIB "grobcov.lib"; '
        f'ring R = (0,{params_str}),({vars_str}),{ordering}; '
        f'ideal F = {gens_str}; '
        f'list C = grobcov(F,"rep",{rep},"ext",{ext},"comment",{comment}); '
        f'C;'
    )
    return cmd


def parse_grobcov_output(text: str) -> List[Dict[str, Any]]:
    """
    Parse the raw output of Singular grobcov into structured segments.

    Each segment represents a region of parameter space where the reduced
    Groebner basis has a fixed leading power product structure.

    Parameters
    ----------
    text : str
        Raw output from Singular grobcov command.

    Returns
    -------
    List[Dict[str, Any]]
        List of segment dictionaries, each containing:
        - 'lpp': List of leading power products (monomials)
        - 'basis': List of reduced basis polynomials
        - 'E': List of equality constraints (polynomials = 0)
        - 'N': List of inequality constraints (polynomials ≠ 0)
        - 'E_p', 'N_p': P-representation guards (raw)
        - 'E_c', 'N_c': C-representation guards (raw)

    Notes
    -----
    The grobcov output has the structure:
    - [i]: Segment i
      - [1]: Leading power products (lpp)
      - [2]: Reduced basis
      - [3]: P-representation segment (nested [1]:[1]: for E, [1]:[2]: for N)
      - [4]: C-representation segment (flat [1]: for E, [2]: for N) [if rep >= 2]

    The P-representation defines the segment as V(E) \\ V(N).
    The C-representation is a cleaner (E, N) pair when available.

    Examples
    --------
    >>> text = '''[1]:
    ...    [1]:
    ...       _[1]=Z01
    ...    [2]:
    ...       _[1]=(Z10+1)*Z01+(Z10*Z11)
    ...    [4]:
    ...       [1]:
    ...          _[1]=0
    ...       [2]:
    ...          _[1]=(Z10+1)'''
    >>> segments = parse_groebner_cover(text)
    >>> len(segments)
    1
    >>> segments[0]['lpp']
    ['Z01']
    >>> segments[0]['N']
    ['Z10+1']
    """
    segments = []
    current = None
    section = None
    sub = None  # 'E' or 'N' inside segment

    for raw in text.splitlines():
        # Count leading spaces to infer nesting
        indent = len(raw) - len(raw.lstrip(' '))
        line = raw.strip()

        # Top-level segment marker: indent 0 and "[n]:"
        if indent == 0 and re.match(r"^\[\d+\]:$", line):
            if current:
                segments.append(current)
            current = {
                'lpp': [], 'basis': [],
                'E_p': [], 'N_p': [],
                'E_c': [], 'N_c': []
            }
            section = None
            sub = None
            continue

        # Section markers inside a segment (indent 3): [1]: lpp, [2]: basis, [3]: P-rep, [4]: C-rep
        if current and indent == 3 and line in ('[1]:', '[2]:', '[3]:', '[4]:'):
            if line == '[1]:':
                section = 'lpp'
                sub = None
            elif line == '[2]:':
                section = 'basis'
                sub = None
            elif line == '[3]:':
                section = 'segment_p'
                sub = None
            else:
                section = 'segment_c'
                sub = None
            continue

        # P-representation: nested structure [3]:[1]:[1]: for E, [3]:[1]:[2]: for N
        # The first [1]: at indent 6 starts the P-rep block, then [1]:/[2]: at indent 9
        if current and section == 'segment_p' and indent == 6 and line == '[1]:':
            # This is the outer wrapper, next level determines E vs N
            continue
        if current and section == 'segment_p' and indent == 9 and line in ('[1]:', '[2]:'):
            sub = 'E' if line == '[1]:' else 'N'
            continue

        # C-representation: flat structure [4]:[1]: for E, [4]:[2]: for N
        if current and section == 'segment_c' and indent == 6 and line in ('[1]:', '[2]:'):
            sub = 'E' if line == '[1]:' else 'N'
            continue

        # Values: lines with _[k]=...
        m = re.search(r"_\[\d+\]\s*=\s*(.*)$", line)
        if m and current is not None:
            val = m.group(1).strip()
            # Strip outer parentheses for lpp/guards, keep for basis
            strip_parens = (section != 'basis')
            if strip_parens and val.startswith('(') and val.endswith(')'):
                val = val[1:-1]

            if section == 'lpp':
                current['lpp'].append(val)
            elif section == 'basis':
                current['basis'].append(val)
            elif section == 'segment_p':
                if sub == 'E':
                    current['E_p'].append(val)
                elif sub == 'N':
                    current['N_p'].append(val)
            elif section == 'segment_c':
                if sub == 'E':
                    current['E_c'].append(val)
                elif sub == 'N':
                    current['N_c'].append(val)

    if current:
        segments.append(current)

    # Normalize guards: prefer C-rep if present, drop trivial entries, deduplicate
    for s in segments:
        E = s['E_c'] if s['E_c'] else s['E_p']
        N = s['N_c'] if s['N_c'] else s['N_p']

        # Filter trivial: '0' in E means no constraint, '1' in N means empty removal
        E = [e for e in E if e not in ('0', '1')]
        N = [n for n in N if n != '1']

        # Deduplicate while preserving order
        seen = set()
        E_clean = []
        for e in E:
            if e not in seen:
                seen.add(e)
                E_clean.append(e)

        seen = set()
        N_clean = []
        for n in N:
            if n not in seen:
                seen.add(n)
                N_clean.append(n)

        s['E'] = E_clean
        s['N'] = N_clean

    return segments


def groebner_cover(
    ideal_gens: Sequence[str],
    solve_vars: Sequence[str],
    param_vars: Sequence[str],
    ordering: str = "dp",
    rep: int = 2,
    ext: int = 1,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    Compute the Groebner Cover of a parametric polynomial ideal.

    The Groebner Cover partitions the parameter space into segments where
    the reduced Groebner basis has a stable leading power product structure.

    Parameters
    ----------
    ideal_gens : Sequence[str]
        List of polynomial generators as strings (in Singular syntax).
    solve_vars : Sequence[str]
        Variables to solve for (appear in the polynomial ring).
    param_vars : Sequence[str]
        Parameter variables (appear in the coefficient field).
    ordering : str, optional
        Monomial ordering for the polynomial ring. Default is "dp".
    rep : int, optional
        Representation type for output (0=P, 1=C, 2=both). Default is 2.
    ext : int, optional
        Extension level for output. Default is 1.
    verbose : bool, optional
        If True, print the raw Singular output. Default is False.

    Returns
    -------
    List[Dict[str, Any]]
        List of segment dictionaries with keys:
        - 'lpp': Leading power products
        - 'basis': Reduced Groebner basis polynomials
        - 'E': Equality constraints defining the segment (p = 0)
        - 'N': Inequality constraints (p ≠ 0)

    Examples
    --------
    >>> # Single equation: Z01*(Z10+1) + Z10*Z11 = 0
    >>> segments = groebner_cover(
    ...     ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
    ...     solve_vars=["Z01"],
    ...     param_vars=["Z10", "Z11"]
    ... )
    >>> len(segments)
    3
    >>> segments[0]['basis']
    ['(Z10+1)*Z01+(Z10*Z11)']
    >>> segments[0]['E']
    []
    >>> segments[0]['N']
    ['Z10+1']

    See Also
    --------
    format_grobcov_input : Format the Singular command string.
    run_grobcov : Run grobcov and return raw output.
    parse_grobcov_output : Parse raw grobcov output.

    Structure
    ---------
    groebner_cover()
    ├── run_grobcov()
    │       └── format_grobcov_input()  → builds command string
    │       └── singular.eval()          → executes in Singular
    └── parse_grobcov_output()          → parses raw output

    Notes
    -----
    This function interfaces with Singular's grobcov.lib library. The grobcov
    algorithm computes a canonical stratification of parameter space such that
    over each stratum, the reduced Groebner basis has constant leading monomials.

    References
    ----------
    A. Montes, M. Wibmer: "Gröbner Bases for Polynomial Systems with Parameters"
    Journal of Symbolic Computation 45 (2010) 1391–1425.
    """
    raw_output = run_grobcov(
        ideal_gens=ideal_gens,
        solve_vars=solve_vars,
        param_vars=param_vars,
        ordering=ordering,
        rep=rep,
        ext=ext,
        verbose=verbose
    )

    segments = parse_grobcov_output(raw_output)
    return segments


def run_grobcov(
    ideal_gens: Sequence[str],
    solve_vars: Sequence[str],
    param_vars: Sequence[str],
    ordering: str = "dp",
    rep: int = 2,
    ext: int = 1,
    verbose: bool = False,
    friendly_levels: bool = False,
) -> str:
    """
    Run grobcov in Singular and return the raw output string.

    This is a lower-level function that executes grobcov and returns
    the unparsed output. Use groebner_cover() for parsed results.

    Parameters
    ----------
    ideal_gens : Sequence[str]
        List of polynomial generators as strings (in Singular syntax).
    solve_vars : Sequence[str]
        Variables to solve for (appear in the polynomial ring).
    param_vars : Sequence[str]
        Parameter variables (appear in the coefficient field).
    ordering : str, optional
        Monomial ordering for the polynomial ring. Default is "dp".
    rep : int, optional
        Representation type for output (0=P, 1=C, 2=both). Default is 2.
    ext : int, optional
        Extension level for output. Default is 1.
    verbose : bool, optional
        If True, print the command and raw output. Default is False.
    friendly_levels : bool, optional
        If True, use Singular's Grob1Levels helper to print the cover in a more
        compact, human-friendly form. Note: this output is not parseable by
        parse_grobcov_output(). Default is False.

    Returns
    -------
    str
        Raw output from Singular grobcov command.

    Examples
    --------
    >>> output = run_grobcov(
    ...     ideal_gens=["Z01*(Z10+1) + Z10*Z11"],
    ...     solve_vars=["Z01"],
    ...     param_vars=["Z10", "Z11"]
    ... )
    >>> "[1]:" in output
    True
    """
    cmd = format_grobcov_input(
        ideal_gens=ideal_gens,
        solve_vars=solve_vars,
        param_vars=param_vars,
        ordering=ordering,
        rep=rep,
        ext=ext
    )

    # Append either the raw cover or the Grob1Levels view.
    if friendly_levels:
        cmd = cmd.replace('C;', 'Grob1Levels(C);')

    if verbose:
        print(f"Singular command:\n{cmd}\n")

    raw_output = singular.eval(cmd)

    if verbose:
        print(f"Raw output:\n{raw_output}\n")

    return raw_output


def summarize_segments(segments: List[Dict[str, Any]]) -> None:
    """
    Print a human-readable summary of Groebner Cover segments.

    Parameters
    ----------
    segments : List[Dict[str, Any]]
        Output from groebner_cover() or parse_groebner_cover().
    """
    print("=" * 80)
    print("GROEBNER COVER SEGMENTS")
    print("=" * 80)

    for i, s in enumerate(segments, 1):
        print(f"\nSegment {i}:")
        print(f"  lpp:   {s['lpp']}")
        print(f"  basis: {s['basis']}")

        guards = []
        if s['E']:
            guards.append("E: " + ", ".join(f"{e} = 0" for e in s['E']))
        if s['N']:
            guards.append("N: " + ", ".join(f"{n} ≠ 0" for n in s['N']))
        print(f"  guards: {'; '.join(guards) if guards else '(none)'}")

        # Quick interpretation
        if '1' in s['lpp'] or '1' in s['basis']:
            print("  → No solutions in this segment")
        elif all(x == '0' for x in s['lpp']) or (not s['basis'] or s['basis'] == ['0']):
            print("  → Variables free subject to equality guards")
        else:
            print("  → Solve triangular basis under guards")
