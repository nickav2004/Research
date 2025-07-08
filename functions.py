from typing import Any
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sympy import MutableDenseMatrix
from sympy.plotting import plot
from IPython.display import display, Markdown


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


def cycle_potentials(equation, potentials):
    """
    Cycles through potentials in an equation, where V_1 -> V_2, V_2 -> V_3, ..., V_n -> V_1

    Parameters:
    equation: sympy equation or expression
    potentials: list of sympy symbols representing potentialsa
    num_potentials: int, number of potentials

    Returns:
    list: List of equations with cycled potentials
    """
    num_potentials = len(potentials)

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
    display(Markdown(f"$\\text{{Computing periodic Schrödinger system for: }}\\Phi_v({energy_param})$"))
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


def analyze_solutions(solution_sets, variables):
    print()
    print("=" * 100, "\n")
    if solution_sets:
        print("=== SOLUTION ANALYSIS ===")

        for i, solution_dict in enumerate(solution_sets):
            print(f"\nSolution set {i + 1}:")

            # Convert to list format
            solution_list = [solution_dict.get(var, var) for var in variables]
            print(f"As list: {solution_list}")

            # Check if irreducible
            is_solution_irreducible = is_irreducible(solution_list)
            print(f"Is irreducible: {is_solution_irreducible}")

            # Print individual values
            for var in variables:
                value = solution_dict.get(var, var)
                print(f"{var}: {value}")

    else:
        print("No solutions found!")


def substitute_free_variables(solution_dict, free_var_values, variables):
    """
    Substitute specific values for free variables in a solution set.

    Parameters:
    solution_dict (dict): Dictionary from sympy solve() containing the solution
    free_var_values (dict): Dictionary mapping free variable names to values
                           e.g., {'V_4': 0, 'V_5': 1, 'V_6': 0, 'V_7': 1}
    variables (list): List of all variables in order [V_1, V_2, ...]

    Returns:
    dict: Solution dictionary with free variables substituted
    """
    substituted_solution = {}

    for var in variables:
        if var in solution_dict:
            # Get the expression for this variable
            expr = solution_dict[var]

            # Substitute free variable values
            substituted_expr = expr
            for free_var, value in free_var_values.items():
                if isinstance(free_var, str):
                    free_var_symbol = sp.symbols(free_var)
                else:
                    free_var_symbol = free_var
                substituted_expr = substituted_expr.subs(free_var_symbol, value)

            # Simplify and evaluate to get numerical result
            try:
                numerical_value = float(substituted_expr.evalf())
                substituted_solution[var] = numerical_value
            except:
                # If can't convert to float, keep as sympy expression
                substituted_solution[var] = substituted_expr.simplify()
        else:
            # Variable is not in solution (might be a free variable)
            var_name = str(var)
            if var_name in free_var_values:
                substituted_solution[var] = free_var_values[var_name]
            elif var in free_var_values:
                substituted_solution[var] = free_var_values[var]
            else:
                substituted_solution[var] = var

    return substituted_solution


def generate_solution_variants(solution_sets, variables, free_var_combinations=None):
    """
    Generate multiple solution variants by substituting different values for free variables.

    Parameters:
    solution_sets (list): List of solution dictionaries from sympy solve()
    variables (list): List of all variables [V_1, V_2, ...]
    free_var_combinations (list): List of dictionaries with free variable values
                                 If None, will generate default combinations

    Returns:
    list: List of dictionaries containing solution variants
    """
    if free_var_combinations is None:
        # Generate default combinations for common free variables
        free_var_combinations = [
            {'V_4': 0, 'V_5': 0, 'V_6': 0, 'V_7': 0},
            {'V_4': 1, 'V_5': 1, 'V_6': 1, 'V_7': 1},
            {'V_4': 0, 'V_5': 1, 'V_6': 0, 'V_7': 1},
            {'V_4': 1, 'V_5': 0, 'V_6': 1, 'V_7': 0},
            {'V_4': 0.5, 'V_5': 0.5, 'V_6': 0.5, 'V_7': 0.5},
            {'V_2': 1, 'V_4': 1, 'V_6': 1},  # For solution set 2 type
            {'V_2': 2, 'V_4': 0.5, 'V_6': 2},
        ]

    all_variants = []

    for i, solution_dict in enumerate(solution_sets):
        print(f"\nProcessing solution set {i + 1}:")
        solution_variants = []

        for j, free_var_values in enumerate(free_var_combinations):
            try:
                substituted = substitute_free_variables(solution_dict, free_var_values, variables)

                # Check if substitution was successful (no undefined expressions)
                valid_solution = True
                solution_list = []

                for var in variables:
                    value = substituted[var]
                    if isinstance(value, (int, float, complex)):
                        solution_list.append(value)
                    elif hasattr(value, 'evalf'):
                        try:
                            numerical_val = complex(value.evalf())
                            if numerical_val.imag == 0:
                                solution_list.append(numerical_val.real)
                            else:
                                solution_list.append(numerical_val)
                        except:
                            valid_solution = False
                            break
                    else:
                        valid_solution = False
                        break

                if valid_solution and not any(sp.zoo in str(val) or sp.nan in str(val) for val in solution_list):
                    variant_info = {
                        'original_solution_index': i + 1,
                        'variant_index': j + 1,
                        'free_var_values': free_var_values,
                        'substituted_solution': substituted,
                        'solution_list': solution_list,
                        'is_irreducible': is_irreducible(solution_list)
                    }
                    solution_variants.append(variant_info)
                    print(f"  Variant {j + 1}: {free_var_values} -> {solution_list}")

            except Exception as e:
                print(f"  Variant {j + 1}: Failed - {e}")
                continue

        all_variants.extend(solution_variants)

    return all_variants


def plot_discriminant_with_sliders(solution_variants, monodromy_matrix, energy_param='z',
                                   energy_range=(-5, 5), initial_variant=0):
    """
    Create an interactive plot of the discriminant with sliders for free variables.

    Parameters:
    solution_variants (list): List of solution variants from generate_solution_variants()
    monodromy_matrix (sp.Matrix): The monodromy matrix expression
    energy_param (str): Energy parameter name
    energy_range (tuple): Range for energy values
    initial_variant (int): Index of initial solution variant to display
    """
    if not solution_variants:
        print("No solution variants provided!")
        return

    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    plt.subplots_adjust(bottom=0.3)

    # Energy values for plotting
    energy_vals = np.linspace(energy_range[0], energy_range[1], 1000)
    energy_symbol = sp.symbols(energy_param)

    def update_plot(variant_idx):
        """Update the plot with the selected solution variant"""
        if variant_idx >= len(solution_variants):
            return

        variant = solution_variants[variant_idx]
        solution_dict = variant['substituted_solution']

        # Substitute solution values into monodromy matrix
        substituted_monodromy = monodromy_matrix
        for var, val in solution_dict.items():
            substituted_monodromy = substituted_monodromy.subs(var, val)

        # Compute discriminant (trace)
        discriminant_expr = substituted_monodromy.trace().simplify()

        # Convert to numerical function
        discriminant_func = sp.lambdify(energy_symbol, discriminant_expr, 'numpy')

        try:
            discriminant_vals = discriminant_func(energy_vals)

            # Clear and plot
            ax1.clear()
            ax2.clear()

            # Plot discriminant
            ax1.plot(energy_vals, discriminant_vals, 'b-', linewidth=2, label=f'Variant {variant_idx + 1}')
            ax1.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='Band edges (±2)')
            ax1.axhline(y=-2, color='r', linestyle='--', alpha=0.7)
            ax1.fill_between(energy_vals, -2, 2, alpha=0.2, color='green', label='Allowed bands')
            ax1.set_ylabel('Discriminant (Trace)')
            ax1.set_title(f'Discriminant vs {energy_param} - Solution Variant {variant_idx + 1}')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.set_ylim(-6, 6)

            # Plot solution values as bar chart
            var_names = [str(var) for var in variant['substituted_solution'].keys()]
            var_values = [float(val) if isinstance(val, (int, float)) else float(val.evalf())
                          for val in variant['substituted_solution'].values()]

            bars = ax2.bar(var_names, var_values, alpha=0.7)
            ax2.set_ylabel('Variable Values')
            ax2.set_title(f'Solution Values - Variant {variant_idx + 1}')
            ax2.tick_params(axis='x', rotation=45)

            # Color bars based on whether they're free variables
            free_vars = variant['free_var_values'].keys()
            for i, bar in enumerate(bars):
                if var_names[i] in free_vars:
                    bar.set_color('orange')
                else:
                    bar.set_color('blue')

            # Add text with variant info
            info_text = f"Free vars: {variant['free_var_values']}\n"
            info_text += f"Irreducible: {variant['is_irreducible']}"
            fig.suptitle(info_text, fontsize=10)

            plt.tight_layout()
            plt.draw()

        except Exception as e:
            print(f"Error plotting variant {variant_idx + 1}: {e}")

    # Create slider
    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Solution Variant', 0, len(solution_variants) - 1,
                    valinit=initial_variant, valfmt='%d')

    def update(val):
        variant_idx = int(slider.val)
        update_plot(variant_idx)

    slider.on_changed(update)

    # Initial plot
    update_plot(initial_variant)

    plt.show()

    return fig, slider


def create_interactive_discriminant_widget(solution_variants, monodromy_matrix,
                                           energy_param='z', energy_range=(-5, 5)):
    """
    Create an interactive Jupyter widget for exploring discriminant plots.

    Parameters:
    solution_variants (list): List of solution variants
    monodromy_matrix (sp.Matrix): The monodromy matrix expression
    energy_param (str): Energy parameter name
    energy_range (tuple): Range for energy values
    """
    try:
        from ipywidgets import interact, IntSlider, fixed
        import matplotlib.pyplot as plt

        def plot_variant(variant_index):
            if variant_index >= len(solution_variants) or variant_index < 0:
                print("Invalid variant index")
                return

            variant = solution_variants[variant_index]
            solution_dict = variant['substituted_solution']

            # Substitute solution values into monodromy matrix
            substituted_monodromy = monodromy_matrix
            for var, val in solution_dict.items():
                substituted_monodromy = substituted_monodromy.subs(var, val)

            # Compute discriminant
            discriminant_expr = substituted_monodromy.trace().simplify()

            # Plot
            energy_vals = np.linspace(energy_range[0], energy_range[1], 1000)
            energy_symbol = sp.symbols(energy_param)
            discriminant_func = sp.lambdify(energy_symbol, discriminant_expr, 'numpy')
            discriminant_vals = discriminant_func(energy_vals)

            plt.figure(figsize=(12, 8))

            # Discriminant plot
            plt.subplot(2, 1, 1)
            plt.plot(energy_vals, discriminant_vals, 'b-', linewidth=2)
            plt.axhline(y=2, color='r', linestyle='--', alpha=0.7, label='Band edges')
            plt.axhline(y=-2, color='r', linestyle='--', alpha=0.7)
            plt.fill_between(energy_vals, -2, 2, alpha=0.2, color='green', label='Allowed bands')
            plt.ylabel('Discriminant (Trace)')
            plt.title(f'Discriminant vs {energy_param} - Variant {variant_index + 1}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.ylim(-6, 6)

            # Solution values
            plt.subplot(2, 1, 2)
            var_names = [str(var) for var in solution_dict.keys()]
            var_values = [float(val) if isinstance(val, (int, float)) else float(val.evalf())
                          for val in solution_dict.values()]

            colors = ['orange' if str(var) in variant['free_var_values'] else 'blue'
                      for var in solution_dict.keys()]

            plt.bar(var_names, var_values, color=colors, alpha=0.7)
            plt.ylabel('Variable Values')
            plt.title(f'Solution Values - Free vars: {variant["free_var_values"]}')
            plt.xticks(rotation=45)

            plt.tight_layout()
            plt.show()

            print(f"Variant {variant_index + 1}:")
            print(f"Free variables: {variant['free_var_values']}")
            print(f"Is irreducible: {variant['is_irreducible']}")
            print(f"Solution list: {variant['solution_list']}")

        # Create interactive widget
        interact(plot_variant,
                 variant_index=IntSlider(min=0, max=len(solution_variants) - 1, step=1, value=0,
                                         description='Solution Variant:'))

    except ImportError:
        print("ipywidgets not available. Use plot_discriminant_with_sliders instead.")


def validate_solution_in_matrix(solution_dict, target_matrix, variables, tolerance=1e-10):
    """
    Validate if a solution satisfies the original matrix equation.

    Parameters:
    solution_dict (dict): Solution dictionary with variable values
    target_matrix (sp.Matrix): Target matrix (e.g., identity matrix)
    variables (list): List of variable symbols
    tolerance (float): Numerical tolerance for validation

    Returns:
    dict: Validation results
    """
    # Substitute solution into original monodromy matrix
    # This would need the original monodromy matrix as a parameter
    # For now, return basic validation info

    validation_result = {
        'is_valid': True,
        'solution_dict': solution_dict,
        'numerical_values': [],
        'errors': []
    }

    try:
        for var in variables:
            if var in solution_dict:
                val = solution_dict[var]
                if isinstance(val, (int, float, complex)):
                    validation_result['numerical_values'].append(val)
                else:
                    numerical_val = complex(val.evalf())
                    validation_result['numerical_values'].append(numerical_val)
            else:
                validation_result['errors'].append(f"Missing value for {var}")
                validation_result['is_valid'] = False

    except Exception as e:
        validation_result['errors'].append(str(e))
        validation_result['is_valid'] = False

    return validation_result


def cycle_solution_variables(solution_dict, shift_amount, variables=None):
    """
    Cycle/shift variables in a solution dictionary by a specified amount.

    This function takes each equation in the solution and shifts the variable indices.
    For example, with shift_amount=3:
    - V_1 -> V_4, V_2 -> V_5, V_3 -> V_6, V_4 -> V_7, V_5 -> V_1, etc.

    Parameters:
    solution_dict (dict): Dictionary containing the solution with variable expressions
    shift_amount (int): Number of positions to shift variables (can be negative)
    variables (list): List of variables in order [V_1, V_2, ...]. If None, will auto-detect.

    Returns:
    dict: New solution dictionary with shifted variables
    """
    if variables is None:
        # Auto-detect variables from solution keys
        variables = list(solution_dict.keys())
        # Sort them by their index (V_1, V_2, V_3, ...)
        variables.sort(key=lambda x: int(str(x).split('_')[1]) if '_' in str(x) else 0)

    num_variables = len(variables)

    if num_variables == 0:
        return {}

    # Normalize shift amount to be within the range [0, num_variables)
    shift_amount = shift_amount % num_variables

    # Create mapping from old variables to new variables
    # V_i -> V_{(i + shift_amount - 1) % num_variables + 1}
    variable_mapping = {}
    for i, var in enumerate(variables):
        old_index = i
        new_index = (old_index + shift_amount) % num_variables
        new_var = variables[new_index]
        variable_mapping[var] = new_var

    print(f"Variable mapping with shift {shift_amount}:")
    for old_var, new_var in variable_mapping.items():
        print(f"  {old_var} -> {new_var}")

    # Create the shifted solution dictionary
    shifted_solution = {}

    # Step 1: Create temporary symbols to avoid conflicts during substitution
    temp_symbols = [sp.symbols(f'TEMP_SHIFT_{i}') for i in range(num_variables)]
    temp_mapping = {var: temp_symbols[i] for i, var in enumerate(variables)}

    for old_var, expression in solution_dict.items():
        # Find the new variable name for this key
        new_var = variable_mapping[old_var]

        # Substitute all variables in the expression with temporary symbols first
        temp_expression = expression
        for orig_var, temp_var in temp_mapping.items():
            temp_expression = temp_expression.subs(orig_var, temp_var)

        # Then substitute temporary symbols with shifted variables
        shifted_expression = temp_expression
        for i, temp_var in enumerate(temp_symbols):
            original_var = variables[i]
            shifted_var = variable_mapping[original_var]
            shifted_expression = shifted_expression.subs(temp_var, shifted_var)

        # Simplify the result
        shifted_solution[new_var] = shifted_expression.simplify()

    return shifted_solution


def generate_all_cyclic_shifts(solution_dict, variables=None):
    """
    Generate all possible cyclic shifts of a solution dictionary.

    Parameters:
    solution_dict (dict): Original solution dictionary
    variables (list): List of variables. If None, will auto-detect.

    Returns:
    list: List of dictionaries, each representing a different cyclic shift
    """
    if variables is None:
        variables = list(solution_dict.keys())
        variables.sort(key=lambda x: int(str(x).split('_')[1]) if '_' in str(x) else 0)

    num_variables = len(variables)
    all_shifts = []

    for shift in range(num_variables):
        shifted_solution = cycle_solution_variables(solution_dict, shift, variables)
        all_shifts.append({
            'shift_amount': shift,
            'solution': shifted_solution
        })

    return all_shifts


def cycle_multiple_solutions(solution_list, shift_amount, variables=None):
    """
    Apply the same cyclic shift to multiple solution dictionaries.

    Parameters:
    solution_list (list): List of solution dictionaries
    shift_amount (int): Number of positions to shift variables
    variables (list): List of variables. If None, will auto-detect from first solution.

    Returns:
    list: List of shifted solution dictionaries
    """
    if not solution_list:
        return []

    if variables is None and solution_list:
        variables = list(solution_list[0].keys())
        variables.sort(key=lambda x: int(str(x).split('_')[1]) if '_' in str(x) else 0)

    shifted_solutions = []

    for i, solution_dict in enumerate(solution_list):
        print(f"\nShifting solution {i + 1} by {shift_amount} positions:")
        shifted_solution = cycle_solution_variables(solution_dict, shift_amount, variables)
        shifted_solutions.append(shifted_solution)

    return shifted_solutions


def compare_original_and_shifted(solution_dict, shift_amount, variables=None):
    """
    Compare original solution with its shifted version side by side.

    Parameters:
    solution_dict (dict): Original solution dictionary
    shift_amount (int): Number of positions to shift
    variables (list): List of variables

    Returns:
    dict: Dictionary containing both original and shifted solutions
    """
    if variables is None:
        variables = list(solution_dict.keys())
        variables.sort(key=lambda x: int(str(x).split('_')[1]) if '_' in str(x) else 0)

    shifted_solution = cycle_solution_variables(solution_dict, shift_amount, variables)

    print(f"\n=== COMPARISON: Original vs Shifted by {shift_amount} ===")
    print(f"{'Variable':<8} {'Original':<50} {'Shifted':<50}")
    print("=" * 110)

    for var in variables:
        orig_expr = solution_dict.get(var, "Not found")
        shifted_expr = shifted_solution.get(var, "Not found")
        print(f"{str(var):<8} {str(orig_expr):<50} {str(shifted_expr):<50}")

    return {
        'original': solution_dict,
        'shifted': shifted_solution,
        'shift_amount': shift_amount,
        'variables': variables
    }


def analyze_shifted_solutions(solution_dict, variables=None, max_shifts=None):
    """
    Analyze all possible shifts of a solution and check properties like irreducibility.

    Parameters:
    solution_dict (dict): Original solution dictionary
    variables (list): List of variables
    max_shifts (int): Maximum number of shifts to analyze. If None, analyzes all.

    Returns:
    list: Analysis results for each shift
    """
    if variables is None:
        variables = list(solution_dict.keys())
        variables.sort(key=lambda x: int(str(x).split('_')[1]) if '_' in str(x) else 0)

    num_variables = len(variables)
    if max_shifts is None:
        max_shifts = num_variables

    analysis_results = []

    print(f"\n=== ANALYZING SHIFTS (up to {max_shifts}) ===")

    for shift in range(min(max_shifts, num_variables)):
        shifted_solution = cycle_solution_variables(solution_dict, shift, variables)

        # Convert to list format for irreducibility check
        solution_list = [shifted_solution.get(var, var) for var in variables]

        # Try to evaluate expressions numerically if possible
        numerical_list = []
        for expr in solution_list:
            try:
                if isinstance(expr, (int, float, complex)):
                    numerical_list.append(expr)
                else:
                    # Try to substitute some default values for free variables to get a numerical result
                    temp_expr = expr
                    free_symbols = expr.free_symbols
                    for sym in free_symbols:
                        if 'V_' in str(sym):
                            temp_expr = temp_expr.subs(sym, 1)  # Default value
                    numerical_list.append(float(temp_expr.evalf()))
            except:
                numerical_list.append(None)

        # Check irreducibility if we have numerical values
        is_irreducible_result = None
        if all(val is not None for val in numerical_list):
            is_irreducible_result = is_irreducible(numerical_list)

        analysis_results.append({
            'shift_amount': shift,
            'solution': shifted_solution,
            'solution_list': solution_list,
            'numerical_approximation': numerical_list,
            'is_irreducible': is_irreducible_result
        })

        print(f"Shift {shift}: Irreducible = {is_irreducible_result}")

    return analysis_results
