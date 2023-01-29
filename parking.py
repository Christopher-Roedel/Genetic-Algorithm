# CJ Roedel Section 1
# Imports
from multiprocessing import Pool
import math
import random
import numpy as np
import scipy as sp
import scipy.interpolate as spi
import matplotlib.pyplot as plt

# Globals
time = 10
cost_tolerance = 0.1
maximum_population_size = 500
maximum_generations = 1200
time_steps = 100
pop_size = 201
num_of_opt_params = 10
bin_code_size = 7
mut_rate = 0.005
K = 200


# Generates gray codes to be used during first generation
def generate_gray_codes():
    n = bin_code_size
    if n <= 0:
        return
    codes = list()
    codes.append('0')
    codes.append('1')
    i = 2
    j = 0
    while True:
        if i >= 1 << n:
            break
        for j in range(i - 1, -1, -1):
            codes.append(codes[j])
        for j in range(i):
            codes[j] = '0' + codes[j]
        for j in range(i, 2 * i):
            codes[j] = '1' + codes[j]
        i = i << 1
    return codes


# Generates the initial populations control variables based on the previously generated gray codes
def generate_first_population():
    populationg = np.arange(pop_size * (2 * num_of_opt_params)).reshape(pop_size, 2 * num_of_opt_params)
    populationg = populationg.astype('<U7')
    for i in range(pop_size):
        for x in range(2 * num_of_opt_params):
            stri = np.random.choice(gray_codes, 1, False)[0]
            populationg[i][x] = stri
    return populationg


# Sets up an array of starting states relative to population size (includes an extra state variable for c)
def set_up_initial_states():
    initial_states = np.arange(pop_size * 5).reshape(pop_size, 5)
    initial_states = initial_states.astype('float64')
    for s in initial_states:
        s[0] = 0
        s[1] = 8
        s[2] = 0
        s[3] = 0
        s[4] = 0
    return initial_states


# decodes the binary controls into floating point controls
def decode():
    population_temp = np.arange(pop_size * (2 * num_of_opt_params)).reshape(pop_size, 2 * num_of_opt_params)
    population_temp = population_temp.astype('float64')
    for i in range(pop_size):
        for x in range(num_of_opt_params * 2):
            if x % 2 == 0:
                population_temp[i][x] = ((int(population_controls[i][x], 2) / (2 ** bin_code_size - 1)) *
                                         (0.524 - -0.524)) + -0.524
            else:
                population_temp[i][x] = ((int(population_controls[i][x], 2) / (2 ** bin_code_size - 1)) * (5 - -5)) + -5
    return population_temp


# Computes the costs of each individual using helper functions
# Also returns the final states, state histories, gamma and beta functions
# so that they can be used at the end in the event that the first successful
# individual is in this generation
def costs(populationi):
    states = set_up_initial_states()
    for p in range(pop_size):
        y = np.arange(num_of_opt_params)
        y = y.astype('float64')
        for i in range(num_of_opt_params):
            y[i] = populationi[p][((i + 1) * 2) - 2]
        x = np.linspace(0, time, num_of_opt_params, True)
        gamma_f = spi.interp1d(x, y, 'cubic')
        for i in range(num_of_opt_params):
            y[i] = populationi[p][(i * 2) + 1]
        beta_f = spi.interp1d(x, y, 'cubic')
        histories = integrate_odes(states, p, gamma_f, beta_f)
    cost_array = compute_cost(states)
    return cost_array, states, histories, gamma_f, beta_f


# integrates the ODES based on the gamma and beta functions as well as the collision checkers
def integrate_odes(states, p, gamma_f, beta_f):
    h = (time / time_steps)
    histories = np.arange(time_steps * 4).reshape(4, time_steps)
    histories = histories.astype('float64')
    histories[0][0] = states[p][0]
    histories[1][0] = states[p][1]
    histories[2][0] = states[p][2]
    histories[3][0] = states[p][3]
    for i in range(time_steps - 1):
        x_temp = histories[0][i]
        y_temp = histories[1][i]
        histories[0][i + 1] = histories[0][i] + ((histories[3][i] * np.cos(histories[2][i])) * h)
        histories[1][i + 1] = histories[1][i] + ((histories[3][i] * np.sin(histories[2][i])) * h)
        histories[2][i + 1] = histories[2][i] + (gamma_f(i * h) * h)
        histories[3][i + 1] = histories[3][i] + (beta_f(i * h) * h)
        if x_temp <= -4 and y_temp <= 3:
            states[p][4] = states[p][4] + (3 - y_temp) ** 2
        elif x_temp >= 4 and y_temp <= 3:
            states[p][4] = states[p][4] + (3 - y_temp) ** 2
        elif y_temp <= -1:
            states[p][4] = states[p][4] + (-1 - y_temp) ** 2
    states[p][0] = histories[0][time_steps - 1]
    states[p][1] = histories[1][time_steps - 1]
    states[p][2] = histories[2][time_steps - 1]
    states[p][3] = histories[3][time_steps - 1]
    return histories


# computes the actual cost of the individuals based on how they performed and if they went out of bounds
def compute_cost(states):
    cost_array = np.arange(pop_size)
    cost_array = cost_array.astype('float64')
    for i in range(pop_size):
        c_temp = states[i][4]
        if c_temp == 0:
            cost_array[i] = math.sqrt((0 - states[i][0]) ** 2 + (0 - states[i][1]) ** 2 + (0 - states[i][2]) ** 2 +
                                      (0 - states[i][3]) ** 2)
        else:
            cost_array[i] = K + c_temp
    return cost_array


# computes the fitness values for each individual
def compute_fitness(cost_array):
    fitness_array = np.arange(pop_size)
    fitness_array = fitness_array.astype('float64')
    for i in range(pop_size):
        fitness_array[i] = 1 / (cost_array[i] + 1)
    return fitness_array


# finds the lowest index given an input of fitness or cost
def find_lowest_index(inpt):
    lowest = inpt[0]
    index = 0
    for i in range(pop_size):
        if inpt[i] < lowest:
            lowest = inpt[i]
            index = i
    return index


# finds the lowest value given fitness or cost
def find_lowest(inpt):
    lowest = inpt[0]
    for i in range(pop_size):
        if inpt[i] < lowest:
            lowest = inpt[i]
    return lowest


# finds the greatest index given fitness or cost
def find_greatest_index(inpt):
    greatest = inpt[0]
    index = 0
    for i in range(pop_size):
        if inpt[i] > greatest:
            greatest = inpt[i]
            index = i
    return index


# creates the next generation of the simulation. Picks 2 parents at a time
# and puts them into an array which is used by the helper function cross_over_mutate
def next_generation(pop_controls, curr_fitness, curr_elite):
    probabilities = [fit / np.sum(curr_fitness) for fit in curr_fitness]
    temp_controls = np.arange(pop_size * (2 * num_of_opt_params)).reshape(pop_size, 2 * num_of_opt_params)
    temp_controls = temp_controls.astype('<U7')
    pop_index = np.arange(pop_size).reshape(pop_size)
    half = int(pop_size / 2)
    for i in range(pop_size):
        pop_index[i] = i
    for i in range(half):
        indx = np.random.choice(pop_index, 2, False, probabilities)
        temp_controls[(i * 2)] = pop_controls[indx[0]]
        temp_controls[(i * 2) + 1] = pop_controls[indx[1]]
    if pop_size % 2 == 1:
        indx = np.random.choice(pop_index, 1, False, probabilities)
        temp_controls[pop_size - 1] = pop_controls[indx]
    new_gen = cross_over_mutate(temp_controls)
    new_gen[0] = curr_elite
    return new_gen


# performs crossover on 2 parents to create 2 children, then applies mutation to the children
def cross_over_mutate(temp_controls):
    finished = np.arange(pop_size * (2 * num_of_opt_params)).reshape(pop_size, 2 * num_of_opt_params)
    finished = finished.astype('<U7')
    half_pop = (pop_size - 1) / 2
    half = int((bin_code_size * num_of_opt_params * 2) / 2)
    for i in range(int(half_pop)):
        tchrome1 = ''
        tchrome2 = ''
        for x in range(num_of_opt_params * 2):
            tchrome1 += temp_controls[(i * 2) + 1][x]
            tchrome2 += temp_controls[(i * 2) + 2][x]
        c1_h1 = tchrome1[:half]
        c1_h2 = tchrome1[half:]
        c2_h1 = tchrome2[:half]
        c2_h2 = tchrome2[half:]
        chrome1 = c1_h1 + c2_h2
        chrome2 = c2_h1 + c1_h2
        mut_chrome1 = ''
        mut_chrome2 = ''
        for z in range(len(chrome1)):
            if random.uniform(0, 1) <= mut_rate:
                if chrome1[z] == '1':
                    mut_chrome1 += '0'
                else:
                    mut_chrome1 += '1'
            else:
                mut_chrome1 += chrome1[z]
        for y in range(len(chrome2)):
            if random.uniform(0, 1) <= mut_rate:
                if chrome2[y] == '1':
                    mut_chrome2 += '0'
                else:
                    mut_chrome2 += '1'
            else:
                mut_chrome2 += chrome2[y]
        for x in range(num_of_opt_params * 2):
            finished[(i * 2) + 1][x] = mut_chrome1[(x * bin_code_size):(len(mut_chrome1) - (len(mut_chrome1)
                                                                              - (x * bin_code_size + bin_code_size)))]
            finished[(i * 2) + 2][x] = mut_chrome2[(x * bin_code_size):(len(mut_chrome1) - (len(mut_chrome1)
                                                                              - (x * bin_code_size + bin_code_size)))]
    if pop_size % 2 == 0:
        finished[pop_size - 1] = temp_controls[pop_size - 1]
    return finished


# the main function which calls the initialization functions before starting the loop
# that will run until a successful individual is found or another break case is met
# then prints out whatever results it has as well as the corresponding graphs
# writes the controls used to a separate file
if __name__ == "__main__":
    gray_codes = generate_gray_codes()
    if pop_size > maximum_population_size:
        pop_size = maximum_population_size
    for gen in range(1200):
        if gen == 0:
            population_controls = generate_first_population()
        else:
            population_controls = next_generation(population_controls, population_fitness, current_elite)
        population_decode = decode()
        population_costs, statess, history, gamma, beta = costs(population_decode)
        J = find_lowest(population_costs)
        population_fitness = compute_fitness(population_costs)
        findex = find_greatest_index(population_fitness)
        current_elite = population_controls[findex]
        print("Generation " + str(gen) + " : J = " + str(J))
        if J <= cost_tolerance:
            break
        if gen == 1199:
            print("Reached max generations")
            break
    print()
    print("Final state values:")
    print("x_f = " + str(statess[findex][0]))
    print("x_y = " + str(statess[findex][1]))
    print("alpha_f = " + str(statess[findex][2]))
    print("v_f = " + str(statess[findex][3]))

    x_values = np.linspace(0, 10, 100, True)
    plt.plot(x_values, history[0], '-')
    plt.xlabel('Time (s)')
    plt.ylabel('x (ft)')
    plt.title('X Over Time')
    plt.show()

    plt.plot(x_values, history[1], '-')
    plt.xlabel('Time (s)')
    plt.ylabel('y (ft)')
    plt.title('Y Over Time')
    plt.show()

    plt.plot(x_values, history[2], '-')
    plt.xlabel('Time (s)')
    plt.ylabel('alpha (rad)')
    plt.title('Alpha Over Time')
    plt.show()

    plt.plot(x_values, history[3], '-')
    plt.xlabel('Time (s)')
    plt.ylabel('v (ft/s)')
    plt.title('Velocity Over Time')
    plt.show()

    plt.plot(x_values, gamma(x_values), '-')
    plt.xlabel('Time (s)')
    plt.ylabel('gamma (rad/s)')
    plt.title('Heading Angle Over Time')
    plt.show()

    plt.plot(x_values, beta(x_values), '-')
    plt.xlabel('Time (s)')
    plt.ylabel('beta (ft/s^2)')
    plt.title('Acceleration Over Time')
    plt.show()

    plt.plot(history[0], history[1], '-')
    plt.xlabel('x (ft)')
    plt.ylabel('y (ft)')
    plt.hlines(3, -15, -4, 'black')
    plt.hlines(3, 4, 15, 'black')
    plt.hlines(-1, -4, 4, 'black')
    plt.vlines(-4, 3, -1, 'black')
    plt.vlines(4, 3, -1, 'black')
    plt.title('Trajectory')
    plt.show()

    f = open("controls.dat", "w")
    for i in range(num_of_opt_params * 2):
        if i % 2 == 0:
            f.write("gamma " + str(i / 2) + ": " + str(population_decode[findex][i]) + "\n")
        else:
            f.write("gamma " + str(int(i / 2)) + ": " + str(population_decode[findex][i]) + "\n")

