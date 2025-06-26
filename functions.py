import sympy as sp
from sympy import MutableDenseMatrix
from sympy.plotting import plot


def create_potential_symbols(period):
    """
    Create symbolic potential variables V_1, V_2, ..., V_{period}

    Parameters:
    period (int): The period of the potential

    Returns:
    list: List of symbolic variables [V_1, V_2, ..., V_{period}]
    """
    return [sp.symbols(f'V_{i + 1}') for i in range(period)]


def create_transfer_matrix(potential_index, v_list, energy_param='z'):
    """
    Create transfer matrix T_n for a discrete Schrödinger operator
    Matrix form: [[energy - v_n, -1], [1, 0]]

    Parameters:
    potential_index (int): Index n for potential V_n
    V_list (list): List of potential symbols
    energy_param (str): Symbol name for energy parameter (default 'z')

    Returns:
    sympy.Matrix: 2x2 transfer matrix
    """
    energy = sp.symbols(energy_param)
    v_n = v_list[potential_index]

    transfer_matrix: MutableDenseMatrix = sp.Matrix([[energy - v_n, -1],
                   [1, 0]])
    return transfer_matrix


def compute_monodromy_matrix(period, energy_param='z'):
    """
    Compute the monodromy matrix M = T_{period} * ... * T_2 * T_1
    for a periodic potential with a given period

    Parameters:
    period (int): Period of the potential
    energy_param (str): Symbol name for energy parameter

    Returns:
    tuple: (monodromy_matrix, potential_symbols)
    """
    # Create potential symbols
    v_list = create_potential_symbols(period)

    print(f"Created potentials: {[str(v) for v in v_list]}")

    # Create transfer matrices
    transfer_matrices = []
    for i in range(period):
        t_i = create_transfer_matrix(i, v_list, energy_param)
        transfer_matrices.append(t_i)

    # Compute monodromy matrix (product in reverse order)
    monodromy = transfer_matrices[0]  # Start with T_1

    for i in range(1, period):
        monodromy = transfer_matrices[i] * monodromy

    # Final simplification
    monodromy_simplified = sp.Matrix([[monodromy[i, j].simplify()
                                       for j in range(2)] for i in range(2)])

    return monodromy_simplified, v_list


def compute_trace_and_discriminant(monodromy_matrix):
    """
    Compute trace and discriminant of monodromy matrix

    Parameters:
    monodromy_matrix: 2x2 sympy Matrix

    Returns:
    tuple: (trace, discriminant)
    """
    trace = monodromy_matrix.trace().simplify()
    discriminant = trace.simplify()  # Discriminant is the trace of the monodromy matrix

    return discriminant


def analyze_spectrum_bands(trace_expr, energy_param='z', z_range=(-5, 5)):
    """
    Analyze and plot the spectrum bands.
    Bands occur where |Trace(M)| ≤ 2

    Parameters:
    trace_expr: Trace expression as a function of energy
    energy_param: Energy parameter symbol name
    z_range: Range for plotting
    """
    z = sp.symbols(energy_param)

    print("Spectrum Analysis:")
    print("Allowed bands occur where |Trace| ≤ 2")
    print("Forbidden gaps occur where |Trace| > 2")

    # Create plot
    p1 = plot(trace_expr, (z, z_range[0], z_range[1]),
              ylim=(-6, 6), show=False,
              title=f'Trace of Monodromy Matrix vs {energy_param}',
              xlabel=energy_param, ylabel='Trace')

    # Add reference lines at ±2 (band edges)
    p2 = plot(2, (z, z_range[0], z_range[1]),
              line_color='red', show=False, label='Band edges')
    p3 = plot(-2, (z, z_range[0], z_range[1]),
              line_color='red', show=False)

    p1.extend(p2)
    p1.extend(p3)
    p1.show()


def cycle_potentials(equation, potentials, num_potentials):
    """
    Cycles through potentials in an equation, where V_1 -> V_2, V_2 -> V_3, ..., V_n -> V_1

    Parameters:
    equation: sympy equation or expression
    potentials: list of sympy symbols representing potentials
    num_potentials: int, number of potentials

    Returns:
    list: List of equations with cycled potentials
    """
    cycled_equations = []

    # Start with the original equation
    for cycle in range(num_potentials):
        if cycle == 0:
            # The first "cycle" is the original equation
            cycled_equations.append(equation)
        else:
            # Create temporary symbols to avoid interference
            temp_symbols = [sp.symbols(f'TEMP_{i}') for i in range(num_potentials)]

            # Step 1: Replace all original potentials with temp symbols
            temp_subs = {potentials[i]: temp_symbols[i] for i in range(num_potentials)}

            # Step 2: Replace temp symbols with cycled potentials
            final_subs = {}
            for i in range(num_potentials):
                new_idx = (i + cycle) % num_potentials
                final_subs[temp_symbols[i]] = potentials[new_idx]

            # Apply substitutions
            if isinstance(equation, sp.Equality):
                temp_eq = sp.Eq(
                    equation.lhs.subs(temp_subs),
                    equation.rhs.subs(temp_subs)
                )
                cycled_eq = sp.Eq(
                    temp_eq.lhs.subs(final_subs),
                    temp_eq.rhs.subs(final_subs)
                )
            else:
                temp_eq = equation.subs(temp_subs)
                cycled_eq = temp_eq.subs(final_subs)

            cycled_equations.append(cycled_eq)

    return cycled_equations


def is_irreducible(potentials):
    """
    Check if a list of potentials is irreducible.
    Now works with both numbers and SymPy expressions/symbols.

    A list is irreducible if all its cyclic shifts are pairwise distinct elements.

    Parameters:
    potentials (list): List of numbers or SymPy expressions representing potential values

    Returns:
    bool: True if the list is irreducible, False if it's reducible
    """
    if not potentials:
        return True  # Empty list is considered irreducible

    period = len(potentials)
    cyclic_shifts = []

    # Generate all cyclic shifts
    for shift in range(period):
        # Create cyclic shift: move the first 'shift' elements to the end
        shifted = potentials[shift:] + potentials[:shift]

        # Convert SymPy expressions to comparable form
        # Use string representation for comparison
        comparable_shift = tuple(str(sp.simplify(item)) for item in shifted)
        cyclic_shifts.append(comparable_shift)

    # Check if all cyclic shifts are pairwise distinct
    unique_shifts = set(cyclic_shifts)

    # If all shifts are distinct, the number of unique shifts equals the period
    return len(unique_shifts) == period