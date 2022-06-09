#!/usr/bin/env python3

from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
from os.path import join

from neural_sampler import NeuralSampler


# sudoku dimensionality

D = 4

# create sudoku weights

def unit_index_to_grid_pos(unit_index, D=D):
    """
    Convert a unit index (i.e. neuron in the neural sampling network) to a grid
    position, return a tuple containing

    * unit index (like input),
    * field index (left to right, top to bottom)
    * grid row (top to bottom),
    * grid column (left to right),
    * block (left to right, top to bottom), and
    * digit encoded by the unit of this index.

    :param unit_index: unit index to convert
    :param D: sudoku dimensionality
    :returns: grid position data tuple
    """
    assert unit_index < D**3
    Dr = int(sqrt(D))
    digit = unit_index % D
    field_id = unit_index // D
    row_id = field_id // D
    col_id = field_id % D
    block_id = (row_id // Dr) * Dr + col_id // Dr

    return unit_index, field_id, row_id, col_id, block_id, digit


def create_sudoku_weight_matrix(D=4, w_inh=-3, bias=2):
    """
    Create a weight matrix for solving D-dimensional sudoku with sampling. Each
    field in the sudoku is encoded by D neurons, one for every possible digit.
    Weights are either 0 (no influence between neurons) or inhibitory.
    Inhibition prevents that conflicting neurons are active at the same time,
    i.e.

    * neurons within

    Note that the weights are independent of the particular sudoku instance,
    which is defined by clamping certain units (forcing them to be active,
    which corresponds to selecting those digits).

    :param D: sudoku dimensionality
    :param w_inh: inhibitory weight for WTA motifs
    :param bias: bias value for all neurons
    :returns: weight matrix (D, D), bias vector (D,)
    """
    assert D in [4, 9]
    assert sqrt(D).is_integer()

    # outer structre is a grid of sqrt(D) * sqrt(D) blocks
    # each block has size sqrt(D) * sqrt(D)
    # each field within a block has D neurons, encoding the chosen digit
    num_units = D**3

    # create logical mapping between neuron index and position in grid

    mapping = np.asarray([unit_index_to_grid_pos(ind, D) for ind in range(num_units)])

    # create weight matrix with inhibitory weight between neurons encoding same
    # digit within a block, row, or column

    W = np.zeros((num_units, num_units))

    for unit_id, field_id, row_id, col_id, block_id, digit in mapping:

        # another digit is chosen in this field
        for i in range(unit_id - digit, unit_id + D - digit):
            if i % D == digit:
                continue

            W[i, unit_id] = w_inh
            W[unit_id, i] = w_inh

        # the same digit is chosen elsewhere within this row
        row_offset = row_id * D ** 2
        row_starting_point = unit_id - field_id * D
        for i in range(row_starting_point + row_offset, row_starting_point + D ** 2 + row_offset, D):
            if i == unit_id:
                continue

            W[i, unit_id] = w_inh
            W[unit_id, i] = w_inh

        # the same digit is chosen elsewhere within this column
        column_starting_point = unit_id - row_id * D**2
        for i in range(column_starting_point, column_starting_point + D**3, D**2):
            if i == unit_id:
                continue

            W[i, unit_id] = w_inh
            W[unit_id, i] = w_inh

        # the same digit is chosen elsewhere within this block
        sqrt_D = int(sqrt(D))
        col_start = col_id - col_id % sqrt_D
        row_start = row_id - row_id % sqrt_D

        for col in range(col_start, col_start + sqrt_D):
            for row in range(row_start, row_start + sqrt_D):
                i = digit + row * D ** 2 + col * D

                if i == unit_id:
                    continue

                W[i, unit_id] = w_inh
                W[unit_id, i] = w_inh

    W[np.arange(num_units), np.arange(num_units)] = 0  # remove autapses

    b = bias * np.ones(num_units)

    return W, b


def check_solution(z, D=D):
    """
    Convert a state from the sampler to a (putative) solution and check whether
    it is valid. In particular:

    * Converting the state from D**3 units (a DxD grid of fields, each with D
    possible values) into an array containing the chosen values. A value is
    chosen for a field if exactly one of the neurons encoding it is active
    (state z = 1). Then, the array contains the filled in digit (a value
    between 1 and D+1). If no units are active for this field, the value should
    be 0, if multiple are active, it should be -1.

    * Check this DxD grid to see whether it is a value solution. This is true
    iff:
      - Every row contains exactly the values [1, D+1].
      - Every column contains exactly the values [1, D+1].
      - Every block contains exactly the values [1, D+1].

    The function returns two values: a bool indicating whether the state
    encodes a valid solution, and the converted DxD array of chosen values (or
    0 / -1 entries).

    :param z: state vector from sampler (D**3,)
    :param D: sudoku dimensionality
    :returns: bool (valid solution), DxD array (chosen values)
    """

    assert z.ndim == 1
    assert z.shape == (D**3,)

    mapping = np.asarray([unit_index_to_grid_pos(ind, D) for ind in range(len(z))])
    solution_array = np.zeros((D, D))
    is_valid = True

    for unit_id, field_id, row_id, col_id, block_id, digit in mapping:

        # another digit is chosen in this field
        digit_indices = []
        for i in range(unit_id - digit, unit_id + D - digit):
            digit_indices.append(z[i])

        sum_digit = sum(digit_indices)

        if sum_digit != 1:
            is_valid = False

        if sum_digit > 1:
            solution_array[row_id, col_id] = -1
        elif sum_digit == 0:
            solution_array[row_id, col_id] = 0
        else:
            solution_array[row_id, col_id] = np.argwhere(digit_indices) + 1

        # the same digit is chosen elsewhere within this row
        sum_digit = 0
        row_offset = row_id * D ** 2
        row_starting_point = unit_id - field_id * D
        for i in range(row_starting_point + row_offset, row_starting_point + D ** 2 + row_offset, D):
            sum_digit += z[i]

        if sum_digit != 1:
            is_valid = False

        # the same digit is chosen elsewhere within this column
        sum_digit = 0
        column_starting_point = unit_id - row_id * D**2
        for i in range(column_starting_point, column_starting_point + D**3, D**2):
            sum_digit += z[i]

        if sum_digit != 1:
            is_valid = False

        # the same digit is chosen elsewhere within this block
        sum_digit = 0
        sqrt_D = int(sqrt(D))
        col_start = col_id - col_id % sqrt_D
        row_start = row_id - row_id % sqrt_D

        for col in range(col_start, col_start + sqrt_D):
            for row in range(row_start, row_start + sqrt_D):
                i = digit + row * D ** 2 + col * D

                sum_digit += z[i]

        if sum_digit != 1:
            is_valid = False

    return is_valid, solution_array


def plot_solution(solution_array, dx=.02, fs=12, textc=None, ax=None):
    """
    Plot a DxD solution array. Fields will contain chosen digits or x if
    multiple neurons were active (if none were active, the field remains
    empty). The grid is plotted into existing axes.

    :param solution_array: DxD solution array, as given by check_solution
    :param dx: border around the sudoku grid, default: 0.02
    :param fs: font size, default: 16
    :param textc: color for text
    :param ax: axes, default: use plt.gca()
    """

    if ax is None:
        ax = plt.gca()

    solution_array = np.asarray(solution_array)
    assert solution_array.ndim == 2
    assert solution_array.shape[0] == solution_array.shape[1]
    D = solution_array.shape[0]
    assert sqrt(D).is_integer()

    Dr = int(sqrt(D))

    ax.set_aspect('equal', adjustable='box')

    seps = np.linspace(0, 1, D + 1)
    mids = (seps[:-1] + seps[1:]) / 2

    for k, sep in enumerate(seps):
        plt.plot([0, 1], [sep, sep], c='k', lw=2 if k in np.arange(0, D + 1, Dr) else .7)
        plt.plot([sep, sep], [0, 1], c='k', lw=2 if k in np.arange(0, D + 1, Dr) else .7)

    for row_id in range(solution_array.shape[0]):
        for col_id in range(solution_array.shape[1]):
            val = solution_array[row_id,col_id]
            if val == 0:
                s = ''
            elif val == -1:
                s = 'x'
            else:
                s = str(val)
            ax.text(mids[col_id], mids[D - 1 - row_id], s, fontsize=fs, color=textc, va='center', ha='center')

    ax.set_xlim([-dx, 1 + dx])
    ax.set_ylim([-dx, 1 + dx])
    ax.axis('off')


def plot_results(states, fn=None):
    """
    Visualize an experiment. Uses check_solution().

    :param states: sampler states
    :param fn: file name for saving the plot
    """
    step = np.arange(len(states))
    solution_data = [check_solution(z) for z in states]

    # check solution uniqueness

    unique_solutions = []

    for is_sol, sol in solution_data:
        if not is_sol:
            continue

        add = True

        for old_sol in unique_solutions:
            if (sol == old_sol).all():
                add = False
                break

        if add:
            unique_solutions += [sol]

    num_solutions = len(unique_solutions)
    max_show = 6

    plt.figure(figsize=(9, 6))
    plt.subplots_adjust(left=.08, top=.95, right=.99, bottom=.09)

    grid_h = [4, 8, 1]
    grid_h0 = [sum(grid_h[:k]) for k in range(len(grid_h))]
    grid = (sum(grid_h), max(1, min(num_solutions, max_show)))

    for k, sol in enumerate(unique_solutions):
        ax = plt.subplot2grid(grid, (grid_h0[0], k), rowspan=grid_h[0])

        if len(unique_solutions) > 1:
            ax.set_title(f'solution #{k+1}')
        else:
            ax.set_title('solution')
        plot_solution(sol, textc=f'C{k}', ax=ax)

        if k + 1 >= max_show:
            print(f'more than {max_show} solutions, skipping some')
            break

    if num_solutions == 0:
        ax = plt.subplot2grid(grid, (grid_h0[0], 0), rowspan=grid_h[0])
        ax.set_title('no solutions found')
        ax.axis('off')

    ax = plt.subplot2grid(grid, (grid_h0[1], 0), rowspan=grid_h[1], colspan=grid[1])
    ax.imshow(states.T, aspect='auto', cmap='gray', interpolation='none')
    ax.set_xticks([])
    ax.set_ylabel('neuron')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax = plt.subplot2grid(grid, (grid_h0[2], 0), rowspan=grid_h[2], colspan=grid[1])
    for k, sol in enumerate(unique_solutions):
        ax.plot(step, [1 if is_sol and (sol == unique_solutions[k]).all() else None for is_sol, sol in solution_data], lw=4, c=f'C{k}')
    ax.set_xlim(min(step), max(step))
    ax.locator_params(axis='x', nbins=4)
    ax.set_yticks([])
    ax.set_xlabel('step')
    ax.set_ylabel('solution\nactive', rotation=0, va='center', ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)

    if fn is not None:
        plt.savefig(fn, dpi=200)


def test_check_solution(solution_array):
    """
    Test the check_solution function by taking a solution_array, creating a
    (fake) sampler state from it, and testing whether check_solution outputs
    the same array. Also returns whether check_solution thinks it is a valid
    state to compare with the expected value.

    :param solution_array: DxD array with solution to test
    :returns: bool (1st output element of check_solution)
    """

    test_z = np.zeros(D**3)

    for row_id in range(solution_array.shape[0]):
        for col_id in range(solution_array.shape[1]):
            val = solution_array[row_id,col_id]
            if val == 0:  # none active
                continue
            elif val == -1:  # multiple active
                val_ = list(range(D))
            else:
                val_ = [solution_array[row_id,col_id] - 1]

            for val in val_:
                ind = D**2 * row_id + D * col_id + val
                test_z[ind] = 1

    valid, sol_recovered = check_solution(test_z)
    assert (sol_recovered == solution_array).all()

    return valid


# test your implementation of check_solution

# a valid solution
debug_solution_1 = np.asarray([
        [4, 1, 3, 2],
        [2, 3, 4, 1],
        [1, 4, 2, 3],
        [3, 2, 1, 4]])

# an invalid solution
debug_solution_2 = np.asarray([
        [4, -1, 3, 2],  # one field with no selection
        [2, 3, 4, 1],
        [1, 4, 2, 3],
        [3, 2, 1, 4]])

# another invalid solution
debug_solution_3 = np.asarray([
        [4, 1, 3, 2],  # multiple 3s / 2s per column
        [2, 3, 4, 1],
        [1, 4, 3, 2],
        [3, 2, 1, 4]])

# these tests will also raise AssertionErrors if check_solution doesn't
# create the same solution array as passed

# test_check_solution(debug_solution_1)
# test_check_solution(debug_solution_2)
# test_check_solution(debug_solution_3)

print('tests ok')

# ----------------------------------------------------------------------
# main function


def main(clamp, title, **kwargs):

    print(f'running {title}')

    outdir = 'out'
    outdir = join(os.path.dirname(__file__), outdir)
    os.makedirs(outdir, exist_ok=True)

    # create weights for solving 4x4 sudoku

    W, b = create_sudoku_weight_matrix(D, **kwargs)

    # plot weights (for debugging)

    plt.figure(figsize=(4, 4))
    plt.imshow(W.T, aspect='auto')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(join(outdir, f'{title}_weights.png'))

    # setup sampler

    sampler = NeuralSampler(W, b)
    sampler.reset(z0=np.zeros(D**3))  # set states to zero
    sampler.clamp_on(clamp)  # clamp units according to sudoku definition

    sampler.run(2000)  # run for 2 s (assuming dt = 1)

    states = sampler.states  # extract states
    assert states[:,clamp].all()  # check that clamping worked

    # plot results

    plot_results(states, fn=join(outdir, f'{title}_results.png'))


# ----------------------------------------------------------------------
# run

sudoku_1 = [
        # (row, col, value)
        (1, 2, 4),
        (2, 1, 4),
        (2, 3, 3),
        (3, 1, 2),
]

sudoku_2 = [
        # (row, col, value)
        (0, 3, 2),
        (2, 0, 1),
        (2, 1, 4),
        (3, 1, 2),
]


sudoku_1 = np.array(sudoku_1)
sudoku_2 = np.array(sudoku_2)

clamp_1 = [sudoku_1[:, 0]*D**2 + sudoku_1[:, 1]*D + sudoku_1[:, 2] - 1][0]
clamp_2 = [sudoku_2[:, 0]*D**2 + sudoku_2[:, 1]*D + sudoku_2[:, 2] - 1][0]

# run main

# main(clamp_1, title='a')
# main(clamp_2, title='b')
main(clamp_2, title='c', bias=4, w_inh=-6)  # 4, -6


plt.show()  # avoid having multiple plt.show()s in your code
