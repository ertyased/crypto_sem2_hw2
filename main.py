import numpy as np


def polynomial_division_with_remainder(dividend, divisor):
    dividend = np.array(dividend, dtype=float)
    divisor = np.array(divisor, dtype=float)
    divisor_degree = len(divisor) - 1

    while len(dividend) >= len(divisor):
        leading_coeff_ratio = dividend[-1] / divisor[-1]
        dividend[-(divisor_degree + 1):] -= leading_coeff_ratio * divisor
        dividend = dividend[:-1]

    return dividend


def get_transformation_matrix(element, irreducible_poly):
    field_degree = len(irreducible_poly) - 1
    transformation_matrix = np.zeros((field_degree, field_degree), dtype=float)

    for i in range(field_degree):
        x_power_i = np.zeros(field_degree, dtype=float)
        x_power_i[i] = 1
        remainder = polynomial_division_with_remainder(np.convolve(x_power_i, element), irreducible_poly)
        remainder = np.pad(remainder, (0, field_degree - len(remainder)), mode='constant')
        transformation_matrix[:, i] = remainder

    return transformation_matrix


def compute_trace_and_norm(element, irreducible_poly):
    transformation_matrix = get_transformation_matrix(element, irreducible_poly)
    trace_value = np.trace(transformation_matrix)
    norm_value = np.linalg.det(transformation_matrix)
    return trace_value, norm_value


f = [1, 0, -2]
coeffs = [1, 1]

norm1, trace1 = compute_trace_and_norm(f, coeffs)
print(f"Норма: {norm1}")
print(f"След: {trace1}")
