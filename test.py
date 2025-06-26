# -*- coding: utf-8 -*-
"""
Discrete Periodic Schrödinger Operators - Research Code
Working with transfer matrices for periodic potentials
"""
from typing import Any
import sympy as sp
from sympy import MutableDenseMatrix
from functions import compute_monodromy_matrix, compute_trace_and_discriminant, is_irreducible


def compute_periodic_schrodinger_system(period: int, energy_param: str = 'z') -> tuple[
    MutableDenseMatrix, object, list[Any]]:
    """
    Compute the monodromy matrix and discriminant for a discrete periodic Schrödinger operator

    Parameters:
    period (int): Period of the potential
    energy_param (str): Energy parameter symbol (default 'z')

    Returns:
    tuple: (monodromy_matrix, discriminant, potential_variables)
    """
    print(f"Computing periodic Schrödinger system")
    print(f"Period: {period}")
    print(f"Energy parameter: {energy_param}")
    print("=" * 50)

    # Compute monodromy matrix
    monodromy, potentials = compute_monodromy_matrix(period, energy_param)

    # Compute discriminant
    discriminant = compute_trace_and_discriminant(monodromy)

    return monodromy, discriminant, potentials


def compute_split_monodromy_identity(period: int, energy_param: str = 'z', sign: int = 1) -> tuple:
    """
    Compute monodromy using the identity M = ±I by splitting transfer matrices.
    Uses the fact that T_1 * T_2 * ... * T_m = ±(T_{m+1} * ... * T_p)^{-1}
    where m = ceil(p/2)

    Parameters:
    period (int): Period of the potential
    energy_param (str): Energy parameter symbol
    sign (int): +1 for M = I, -1 for M = -I

    Returns:
    tuple: (left_product, right_product_inverse, potential_variables)
    """
    from functions import create_potential_symbols, create_transfer_matrix
    import math

    # Create potential symbols
    v_list = create_potential_symbols(period)

    # Split point - use ceiling to handle odd periods
    m = math.ceil(period / 2)

    # Compute left product: T_1 * T_2 * ... * T_m
    left_product = sp.eye(2)
    for i in range(m):
        t_i = create_transfer_matrix(i, v_list, energy_param)
        left_product = t_i * left_product

    # Compute right product: T_{m+1} * ... * T_p
    right_product = sp.eye(2)
    for i in range(m, period):
        t_i = create_transfer_matrix(i, v_list, energy_param)
        right_product = t_i * right_product

    # Compute inverse of the right product
    right_product_inverse = right_product.inv()

    return left_product, sign * right_product_inverse, v_list


def get_monodromy_identity_equations_split(period: int, energy_param: str = 'z', sign: int = 1) -> list:
    """
    Get simplified equations using split monodromy identity.
    Returns equations of the form: left_product - sign * right_product_inverse = 0

    Parameters:
    period (int): Period of the potential
    energy_param (str): Energy parameter symbol
    sign (int): +1 for M = I, -1 for M = -I

    Returns:
    list: List of simplified equations (one for each matrix entry)
    """
    left_product, right_product_inverse, v_list = compute_split_monodromy_identity(period, energy_param, sign)

    equations = []
    # Generate equations for each matrix entry: left[i,j] - right[i,j] = 0
    for i in range(2):
        for j in range(2):
            eq = sp.Eq(left_product[i, j] - right_product_inverse[i, j], 0)
            equations.append(eq.simplify())

    return equations



def find_closed_gaps_minimal_equations(period: int) -> dict:
    """
    Find the maximum number of closed gaps using split monodromy identity for efficiency.
    Uses pigeonhole principle to distribute equations across energy values optimally.

    Parameters:
    period (int): Period of the potential

    Returns:
    dict: Dictionary with gap count as key and solutions as value
    """
    import math
    from functions import create_potential_symbols, cycle_potentials

    v_list = create_potential_symbols(period)
    results = {}

    # Check up to period-2 gaps (Jacobi theorem constraint)
    max_gaps = period - 2

    for num_gaps in range(1, max_gaps + 1):
        print(f"\nSearching for {num_gaps} closed gap(s)...")

        # Create energy symbols for each assumed gap
        energy_symbols = [sp.symbols(f'E_{i + 1}') for i in range(num_gaps)]

        all_equations = []

        # Generate split monodromy equations for each energy
        split_equations_per_energy = []
        for i, energy in enumerate(energy_symbols):
            # Get split monodromy identity equations (much simpler than full monodromy)
            split_eqs = get_monodromy_identity_equations_split(period, str(energy), sign=1)
            split_equations_per_energy.append(split_eqs)

        # Use pigeonhole principle to distribute equations optimally
        equations_needed = len(v_list)
        equations_per_energy = math.ceil(equations_needed / num_gaps)

        # Distribute equations across energies using round-robin
        equation_index = 0
        for round_num in range(equations_per_energy):
            for energy_idx in range(num_gaps):
                if len(all_equations) >= equations_needed:
                    break
                if round_num < len(split_equations_per_energy[energy_idx]):
                    all_equations.append(split_equations_per_energy[energy_idx][round_num])
            if len(all_equations) >= equations_needed:
                break

        # If still not enough equations, use cyclic shifts
        if len(all_equations) < equations_needed:
            original_equations = all_equations.copy()
            for eq in original_equations:
                if len(all_equations) >= equations_needed:
                    break
                cycled = cycle_potentials(eq, v_list, period)
                # Add cycled equations (skip the original)
                for cycled_eq in cycled[1:]:
                    all_equations.append(cycled_eq)
                    if len(all_equations) >= equations_needed:
                        break

        # Select exactly the number of equations needed
        selected_equations = all_equations[:equations_needed]

        print(f"Using {len(selected_equations)} equations for {len(v_list)} variables")
        print(f"Equations distributed: {equations_per_energy} per energy (round-robin)")

        # Try to solve the system with timeout and faster methods
        try:
            # Use faster solving method if available
            solutions = solve_system_optimized(selected_equations, v_list)

            if solutions:
                # Filter for irreducible solutions
                irreducible_solutions = []
                for sol in solutions:
                    sol_list = [sol.get(var, var) for var in v_list]
                    if is_irreducible(sol_list):
                        irreducible_solutions.append(sol)

                if irreducible_solutions:
                    results[num_gaps] = {
                        'solutions': irreducible_solutions,
                        'energy_params': energy_symbols
                    }
                    print(f"Found {len(irreducible_solutions)} irreducible solutions for {num_gaps} gaps")
                else:
                    print(f"No irreducible solutions found for {num_gaps} gaps")
                    results[num_gaps] = {'solutions': [], 'energy_params': energy_symbols}
                    print(f"Stopping search as no valid solutions found for {num_gaps} gaps")
                    break
            else:
                print(f"No solutions found for {num_gaps} gaps")
                results[num_gaps] = {'solutions': [], 'energy_params': energy_symbols}
                print(f"Stopping search as no solutions found for {num_gaps} gaps")
                break

        except Exception as e:
            print(f"Error solving for {num_gaps} gaps: {e}")
            results[num_gaps] = {'solutions': [], 'energy_params': energy_symbols}
            print(f"Stopping search due to error for {num_gaps} gaps")
            break

    return results


def solve_system_optimized(equations, variables, timeout=300):
    """
    Solve system with optimization techniques and timeout.

    Parameters:
    equations (list): List of equations to solve
    variables (list): Variables to solve for
    timeout (int): Timeout in seconds

    Returns:
    list: Solutions or empty list if timeout/error
    """
    import signal
    from contextlib import contextmanager

    @contextmanager
    def time_limit(seconds):
        def signal_handler(signum, frame):
            raise TimeoutError("Timed out!")

        signal.signal(signal.SIGALRM, signal_handler)
        signal.alarm(seconds)
        try:
            yield
        finally:
            signal.alarm(0)

    try:
        # Try with time limit
        with time_limit(timeout):
            print("Attempting to solve system with SymPy...")

            # Try different SymPy solve methods
            try:
                # Method 1: Standard solve
                solutions = sp.solve(equations, variables, dict=True)
                if solutions:
                    return solutions
            except:
                pass

            try:
                # Method 2: Solve with manual=True for more control
                solutions = sp.solve(equations, variables, dict=True, manual=True)
                if solutions:
                    return solutions
            except:
                pass

            try:
                # Method 3: Use nonlinsolve for nonlinear systems
                solutions = sp.nonlinsolve(equations, variables)
                if solutions:
                    # Convert to dict format
                    result = []
                    for sol in solutions:
                        sol_dict = {var: val for var, val in zip(variables, sol)}
                        result.append(sol_dict)
                    return result
            except:
                pass

            return []

    except TimeoutError:
        print(f"Solve operation timed out after {timeout} seconds")
        return []
    except Exception as e:
        print(f"Error in optimized solver: {e}")
        return []


def solve_numerically_if_needed(equations, variables, sample_energies=None):
    """
    Alternative numerical approach when symbolic solving is too slow.

    Parameters:
    equations (list): System of equations
    variables (list): Variables to solve for
    sample_energies (list): Sample energy values to try

    Returns:
    list: Numerical solutions
    """
    if sample_energies is None:
        sample_energies = [0, 1, -1, 2, -2, 0.5, -0.5]

    import numpy as np
    from scipy.optimize import fsolve

    print("Attempting numerical solution...")

    # Convert to numerical functions
    try:
        # Substitute sample energy values and try to solve numerically
        for energy_val in sample_energies:
            print(f"Trying energy = {energy_val}")

            # Substitute energy values in equations
            num_equations = []
            for eq in equations:
                # Find energy symbols and substitute
                energy_symbols = [sym for sym in eq.free_symbols if 'E_' in str(sym)]
                substituted_eq = eq
                for e_sym in energy_symbols:
                    substituted_eq = substituted_eq.subs(e_sym, energy_val)
                num_equations.append(substituted_eq)

            # Try to solve the substituted system
            try:
                solutions = sp.solve(num_equations, variables, dict=True)
                if solutions:
                    print(f"Found numerical solutions at energy = {energy_val}")
                    return solutions
            except:
                continue

        print("No numerical solutions found")
        return []

    except Exception as e:
        print(f"Error in numerical solver: {e}")
        return []


def apply_cyclic_shift_monodromy_identity(monodromy_original: sp.Matrix, v_list: list,
                                          energy_param: str = 'z') -> sp.Matrix:
    """
    Apply the cyclic shift identity for monodromy matrices:
    M_shifted = T_1 * M_original * T_1^{-1}
    where T_1 is the transfer matrix with potential v_1

    Parameters:
    monodromy_original (sp.Matrix): Original monodromy matrix
    v_list (list): List of potential symbols
    energy_param (str): Energy parameter symbol

    Returns:
    sp.Matrix: Monodromy matrix for the cyclically shifted potential
    """
    from functions import create_transfer_matrix

    # Create transfer matrix for first potential
    t_1 = create_transfer_matrix(0, v_list, energy_param)
    t_1_inv = t_1.inv()

    # Apply the identity: M_shifted = T_1 * M_original * T_1^{-1}
    monodromy_shifted = t_1 * monodromy_original * t_1_inv

    return monodromy_shifted.simplify()


def analyze_closed_gaps_comprehensive(period: int) -> dict:
    """
    Comprehensive analysis of closed gaps.

    Parameters:
    period (int): Period of the potential

    Returns:
    dict: Complete analysis results
    """
    print(f"=== COMPREHENSIVE CLOSED GAP ANALYSIS ===")
    print(f"Period: {period}")
    print("=" * 50)

    # Find solutions using minimal equations
    gap_results = find_closed_gaps_minimal_equations(period)

    # Analyze each result
    analysis = {}
    for num_gaps, result_data in gap_results.items():
        # Extract solutions and energy parameters from the result dictionary
        solutions = result_data.get('solutions', [])
        energy_params = result_data.get('energy_params', [])

        analysis[num_gaps] = {
            'solutions': solutions,
            'energy_params': energy_params,
            'count': len(solutions),
            'irreducible_count': len(solutions)  # Already filtered for irreducible
        }

        # Display solutions properly
        print(f"\n{num_gaps} closed gap(s): {len(solutions)} solutions found")
        print(f"Energy parameters: {[str(e) for e in energy_params]}")

        for i, sol in enumerate(solutions):
            print(f"  Solution {i + 1}:")
            for var, val in sol.items():
                print(f"    {var}: {val}")

    return analysis


def main():
    """
    Main function demonstrating the flexible monodromy analysis
    """
    period = 5

    # Example 1: Basic monodromy computation
    print("=== EXAMPLE 1: Basic Monodromy Computation ===")
    monodromy, discriminant, potentials = compute_periodic_schrodinger_system(period)

    # Example 2: Using split monodromy identity
    print("\n=== EXAMPLE 2: Split Monodromy Identity ===")
    left_prod, right_prod_inv, _ = compute_split_monodromy_identity(period, sign=1)
    print("Left product (first half):")
    sp.pprint(left_prod)
    print("\nRight product inverse (second half):")
    sp.pprint(right_prod_inv)

    # Example 3: Find closed gaps
    print("\n=== EXAMPLE 3: Closed Gap Analysis ===")
    results = analyze_closed_gaps_comprehensive(period)



if __name__ == "__main__":
    main()