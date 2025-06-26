# -*- coding: utf-8 -*-
"""
Discrete Periodic Schrödinger Operators - Research Code
Working with transfer matrices for periodic potentials
"""
from typing import Any

import sympy as sp
from sympy import MutableDenseMatrix
from functions import compute_monodromy_matrix, compute_trace_and_discriminant, is_irreducible


def main(period: int, energy_param: str = 'z') -> tuple[MutableDenseMatrix, object, list[Any]]:
    """
    Main function to analyze discrete periodic Schrödinger operator

    Parameters:
    period (int): Period of the potential (default 3)
    energy_param (str): Energy parameter symbol (default 'z')
    """
    print(f"Analyzing discrete periodic Schrödinger operator")
    print(f"Period: {period}")
    print(f"Energy parameter: {energy_param}")
    print("=" * 50)

    # Compute monodromy matrix
    monodromy, potentials = compute_monodromy_matrix(period, energy_param)

    # Compute discriminant
    discriminant = compute_trace_and_discriminant(monodromy)

    # print("Discriminant (Trace of Monodromy Matrix):")
    # sp.pprint(discriminant)
    # print()

    return monodromy, discriminant, potentials


if __name__ == "__main__":
    monodromy_matrix, discriminant_expr, potential_var = main(period=5)
    monodromy_matrix_2, discriminant_expr_2, potential_var_2 = main(period=6, energy_param='y')

    # Monodromy Matrix entries
    # print()
    # for i in range(4):
    #     print(f"Monodromy Matrix Entry: {i + 1}")
    #     sp.pprint(monodromy_matrix[i])
    #     print()

    E = sp.symbols('E')
    z = sp.symbols('z')
    # equations = cycle_potentials(sp.Eq(monodromy_matrix[3], 1), potential_var, len(potential_var))
    equations = [sp.Eq(monodromy_matrix[0], 1), sp.Eq(monodromy_matrix[1], 0), sp.Eq(monodromy_matrix[2], 0),
                 sp.Eq(monodromy_matrix[3], 1), sp.Eq(monodromy_matrix_2[0], 1), sp.Eq(monodromy_matrix_2[1], 0),
                 sp.Eq(monodromy_matrix_2[2], 0), sp.Eq(monodromy_matrix_2[3], 1)]

    # for i, eq in enumerate(equations):
    #     equations[i] = eq.subs(z, 0)

    solutions = sp.solve(equations, potential_var, dict=True)

    for solution in solutions:
        for key, value in solution.items():
            print(f"{key}: {value}")
        print()

    if solutions:
        print("=== SOLUTION ANALYSIS ===")

        for i, solution_dict in enumerate(solutions):
            print(f"\nSolution set {i + 1}:")

            # Convert to list format
            solution_list = [solution_dict.get(var, var) for var in potential_var]
            print(f"As list: {solution_list}")

            # Check if irreducible
            is_solution_irreducible = is_irreducible(solution_list)
            print(f"Is irreducible: {is_solution_irreducible}")

            # Print individual values
            for var in potential_var:
                value = solution_dict.get(var, var)
                print(f"{var}: {value}")
    else:
        print("No solutions found!")
