import numpy as np
import operator


def gaussian_prob(x, para_tuple):
    """Compute the probability of a given x value

    Args:
        x (float): observation value
        para_tuple (tuple): contains two elements, (mean, standard deviation)

    Return:
        Probability of seeing a value "x" in a Gaussian distribution.

    Note:
        We simplify the problem so you don't have to take care of integrals.
        Theoretically speaking, the returned value is not a probability of x,
        since the probability of any single value x from a continuous 
        distribution should be zero, instead of the number outputed here.
        By definition, the Gaussian percentile of a given value "x"
        is computed based on the "area" under the curve, from left-most to x. 
        The probability of getting value "x" is zero because a single value "x"
        has zero width, however, the probability of a range of value can be
        computed, for say, from "x - 0.1" to "x + 0.1".

    """
    if para_tuple == (None, None):
        return 0.0

    mean, std = para_tuple
    gaussian_percentile = (2 * np.pi * std**2)**-0.5 * \
                          np.exp(-(x - mean)**2 / (2 * std**2))
    return gaussian_percentile


def part_1_a():
    """Provide probabilities for the word HMMs outlined below.

    Word BUY, CAR, and HOUSE.

    Review Udacity Lesson 8 - Video #29. HMM Training

    Returns:
        tuple() of
        (prior probabilities for all states for word BUY,
         transition probabilities between states for word BUY,
         emission parameters tuple(mean, std) for all states for word BUY,
         prior probabilities for all states for word CAR,
         transition probabilities between states for word CAR,
         emission parameters tuple(mean, std) for all states for word CAR,
         prior probabilities for all states for word HOUSE,
         transition probabilities between states for word HOUSE,
         emission parameters tuple(mean, std) for all states for word HOUSE,)


        Sample Format (not complete):
        (
            {'B1': prob_of_starting_in_B1, 'B2': prob_of_starting_in_B2, ...},
            {'B1': {'B1': prob_of_transition_from_B1_to_B1,
                    'B2': prob_of_transition_from_B1_to_B2,
                    'B3': prob_of_transition_from_B1_to_B3,
                    'Bend': prob_of_transition_from_B1_to_Bend},
             'B2': {...}, ...},
            {'B1': tuple(mean_of_B1, standard_deviation_of_B1),
             'B2': tuple(mean_of_B2, standard_deviation_of_B2), ...},
            {'C1': prob_of_starting_in_C1, 'C2': prob_of_starting_in_C2, ...},
            {'C1': {'C1': prob_of_transition_from_C1_to_C1,
                    'C2': prob_of_transition_from_C1_to_C2,
                    'C3': prob_of_transition_from_C1_to_C3,
                    'Cend': prob_of_transition_from_C1_to_Cend},
             'C2': {...}, ...}
            {'C1': tuple(mean_of_C1, standard_deviation_of_C1),
             'C2': tuple(mean_of_C2, standard_deviation_of_C2), ...}
            {'H1': prob_of_starting_in_H1, 'H2': prob_of_starting_in_H2, ...},
            {'H1': {'H1': prob_of_transition_from_H1_to_H1,
                    'H2': prob_of_transition_from_H1_to_H2,
                    'H3': prob_of_transition_from_H1_to_H3,
                    'Hend': prob_of_transition_from_H1_to_Hend},
             'H2': {...}, ...}
            {'H1': tuple(mean_of_H1, standard_deviation_of_H1),
             'H2': tuple(mean_of_H2, standard_deviation_of_H2), ...}
        )
    """

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    b_transition_probs = {
        'B1': {'B1': 0.625, 'B2': 0.375, 'B3': 0.000, 'Bend': 0.000},
        'B2': {'B1': 0.000, 'B2': 0.625, 'B3': 0.375, 'Bend': 0.000},
        'B3': {'B1': 0.000, 'B2': 0.000, 'B3': 0.625, 'Bend': 0.375},
        'Bend': {'B1': 0.000, 'B2': 0.000, 'B3': 0.000, 'Bend': 1.000},
    }
    # Parameters for end state is not required
    b_emission_paras = {
        'B1': (41.750, 2.773), 
        'B2': (58.625, 5.678),
        'B3': (53.125, 5.418),
        'Bend': (None, None)
    }

    """Word CAR"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.000,
        'C3': 0.000,
        'Cend': 0.000,
    }
    c_transition_probs = {
        'C1': {'C1': 0.667, 'C2': 0.333, 'C3': 0.000, 'Cend': 0.000},
        'C2': {'C1': 0.000, 'C2': 0.000, 'C3': 1.000, 'Cend': 0.000},
        'C3': {'C1': 0.000, 'C2': 0.000, 'C3': 0.800, 'Cend': 0.200},
        'Cend': {'C1': 0.000, 'C2': 0.000, 'C3': 0.000, 'Cend': 1.000},
    }
    # Parameters for end state is not required
    c_emission_paras = {
        'C1': (35.667, 4.899),
        'C2': (43.667, 1.700),
        'C3': (44.200, 7.341),
        'Cend': (None, None)
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.000,
        'H3': 0.000,
        'Hend': 0.000,
    }
    # Probability of a state changing to another state.
    h_transition_probs = {
        'H1': {'H1': 0.667, 'H2': 0.333, 'H3': 0.000, 'Hend': 0.000},
        'H2': {'H1': 0.000, 'H2': 0.857, 'H3': 0.143, 'Hend': 0.000},
        'H3': {'H1': 0.000, 'H2': 0.000, 'H3': 0.812, 'Hend': 0.188},
        'Hend': {'H1': 0.000, 'H2': 0.000, 'H3': 0.000, 'Hend': 1.000},
    }
    # Parameters for end state is not required
    h_emission_paras = {
        'H1': (45.333, 3.972),
        'H2': (34.952, 8.127),
        'H3': (67.438, 5.733),
        'Hend': (None, None)
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def viterbi(evidence_vector, states, prior_probs,
            transition_probs, emission_paras):
    """Viterbi Algorithm to calculate the most likely states give the evidence.

    Args:
        evidence_vector (list): List of right hand Y-axis positions (integer).

        states (list): List of all states in a word. No transition between words.
                       example: ['B1', 'B2', 'B3', 'Bend']

        prior_probs (dict): prior distribution for each state.
                            example: {'X1': 0.25,
                                      'X2': 0.25,
                                      'X3': 0.25,
                                      'Xend': 0.25}

        transition_probs (dict): dictionary representing transitions from each
                                 state to every other state.

        emission_paras (dict): parameters of Gaussian distribution 
                                from each state.

    Return:
        tuple of
        ( A list of states the most likely explains the evidence,
          probability this state sequence fits the evidence as a float )

    Note:
        You are required to use the function gaussian_prob to compute the
        emission probabilities.
    """
    paths_list = []
    answer = ['B1', 'B2', 'B2', 'B2', 'B2', 'B2']
    if len(evidence_vector) > 0:

        # Store path in list of dictionaries {previous path, current path}
        b_init = (['B1'], gaussian_prob(evidence_vector[0], emission_paras['B1']) * prior_probs['B1'])
        c_init = (['C1'], gaussian_prob(evidence_vector[0], emission_paras['C1']) * prior_probs['C1'])
        h_init = (['H1'], gaussian_prob(evidence_vector[0], emission_paras['H1']) * prior_probs['H1'])
        paths_list = [b_init, c_init, h_init]

        # Initailize first step since evidence must start in state 1

        for i in range(1,len(evidence_vector)):

            j = 0
            end = len(paths_list)

            for path in paths_list:

                if j > end:
                    break

                # Determine which state list to use
                if path[0][0] == 'B1':
                    t_states = ['B1', 'B2', 'B3']
                elif path[0][0] == 'C1':
                    t_states = ['C1', 'C2', 'C3']
                else:
                    t_states = ['H1', 'H2', 'H3']

                if len(path[0]) == i:
                    last_state = path[0][-1]

                    for state in t_states:

                        if transition_probs[last_state][state] != 0:
                            prior_prob = path[1]
                            trans_prob = transition_probs[last_state][state]
                            current_prob = gaussian_prob(evidence_vector[i], emission_paras[state]) * prior_prob * trans_prob
                            current_path = path[0][:]
                            current_path.append(state)
                            paths_list.append((current_path, current_prob))

                            #print(current_path, current_prob, trans_prob, prior_prob)

                j = j + 1

    probability = 0
    sequence = None
    for path in paths_list:

        if len(path[0]) == len(evidence_vector):
            if path[1] > probability:
                probability = path[1]
                sequence = path[0]

    #print(probability, sequence)
    return sequence, probability

def part_2_a():
    """Provide probabilities for the word HMMs outlined below.

    Now, at each time frame you are given with 2 observations (right hand Y
    position & left hand Y position). Use the result you derived in
    part_1_a, accompany with the provided probability for left hand, create
    a tuple of (right-y, left-y) to represent high-dimention transition & 
    emission probabilities.
    """

    """Word BUY"""
    b_prior_probs = {
        'B1': 0.333,
        'B2': 0.000,
        'B3': 0.000,
        'Bend': 0.000,
    }
    # example: {'B1': {'B1' : (right-hand Y, left-hand Y), ... }
    b_transition_probs = {
        'B1': {'B1': (0.625, 0.700), 'B2': (0.375, 0.300), 'B3': (0.000, 0.000), 'Bend': (0.000, 0.000)},
        'B2': {'B1': (0.000, 0.000), 'B2': (0.625, 0.050), 'B3': (0.375, 0.950), 'Bend': (0.000, 0.000)},
        'B3': {'B1': (0.000, 0.000), 'B2': (0.000, 0.000), 'B3': (0.625, 0.727), 'Bend': (0.125, 0.091), 'H1': (0.125, 0.091), 'C1': (0.125, 0.091)},
        'Bend': {'B1': (0.000, 0.000), 'B2': (0.000, 0.000), 'B3': (0.000, 0.000), 'Bend': (1.000, 1.000)},
    }
    # example: {'B1': [(right-mean, right-std), (left-mean, left-std)] ...}
    b_emission_paras = {
        'B1': [(41.750, 2.773), (108.200, 17.314)],
        'B2': [(58.625, 5.678), (78.670, 1.886)],
        'B3': [(53.125, 5.418), (64.182, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    """Word Car"""
    c_prior_probs = {
        'C1': 0.333,
        'C2': 0.,
        'C3': 0.,
        'Cend': 0.,
    }
    c_transition_probs = {
        'C1': {'C1': (0.667, 0.700), 'C2': (0.333, 0.300), 'C3': (0., 0.), 'Cend': (0., 0.)},
        'C2': {'C1': (0., 0.), 'C2': (0., 0.625), 'C3': (1.000, 0.375), 'Cend': (0., 0.)},
        'C3': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0.800, 0.625), 'Cend': (0.067, 0.125), 'B1': (0.067, 0.125), 'H1': (0.067, 0.125)},
        'Cend': {'C1': (0., 0.), 'C2': (0., 0.), 'C3': (0., 0.), 'Cend': (1.000, 1.000)},
    }
    c_emission_paras = {
        'C1': [(35.667, 4.899), (56.300, 10.659)],
        'C2': [(43.667, 1.700), (37.110, 4.306)],
        'C3': [(44.200, 7.341), (50.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    """Word HOUSE"""
    h_prior_probs = {
        'H1': 0.333,
        'H2': 0.,
        'H3': 0.,
        'Hend': 0.,
    }
    h_transition_probs = {
        'H1': {'H1': (0.667, 0.700), 'H2': (0.333, 0.300), 'H3': (0., 0.), 'Hend': (0., 0.)},
        'H2': {'H1': (0., 0.), 'H2': (0.857, 0.842), 'H3': (0.143, 0.158), 'Hend': (0., 0.)},
        'H3': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0.812, 0.824), 'Hend': (0.063, 0.059), 'B1': (0.063, 0.059), 'C1': (0.063, 0.059)},
        'Hend': {'H1': (0., 0.), 'H2': (0., 0.), 'H3': (0., 0.), 'Hend': (1.000, 1.000)},
    }
    h_emission_paras = {
        'H1': [(45.333, 3.972), (53.600, 7.392)],
        'H2': [(34.952, 8.127), (37.168, 8.875)],
        'H3': [(67.438, 5.733), (74.176, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_prior_probs, b_transition_probs, b_emission_paras,
            c_prior_probs, c_transition_probs, c_emission_paras,
            h_prior_probs, h_transition_probs, h_emission_paras,)


def multidimensional_viterbi(evidence_vector, states, prior_probs,
                             transition_probs, emission_paras):
    """Decode the most likely word phrases generated by the evidence vector.

    States, prior_probs, transition_probs, and emission_probs will now contain
    all the words from part_2_a.
    """
    paths_list = []

    if len(evidence_vector) > 0:

        # Store path in list of dictionaries {previous path, current path}
        b_init = (['B1'], gaussian_prob(evidence_vector[0][0], emission_paras['B1'][0]) * gaussian_prob(evidence_vector[0][1], emission_paras['B1'][1]) * prior_probs['B1'])
        c_init = (['C1'], gaussian_prob(evidence_vector[0][0], emission_paras['C1'][0]) * gaussian_prob(evidence_vector[0][1], emission_paras['C1'][1]) * prior_probs['C1'])
        h_init = (['H1'], gaussian_prob(evidence_vector[0][0], emission_paras['H1'][0]) * gaussian_prob(evidence_vector[0][1], emission_paras['H1'][1]) * prior_probs['H1'])
        paths_list = [b_init, c_init, h_init]

        # Initailize first step since evidence must start in state 1
        answer = ['B1', 'B1', 'B2', 'B2', 'B3', 'B3', 'B3']
        values = []
        p25 = 0
        for i in range(1, len(evidence_vector)):
            j = 0
            end = len(paths_list)

            for path in paths_list:

                if j > end:
                    break
                
                if len(path[0]) == i and path[1] > p25:
                    last_state = path[0][-1]

                    # Determine which state list to use
                    if path[0][-1] in ('B1', 'B2', 'B3'):
                        t_states = ['B1', 'B2', 'B3', 'C1', 'H1']
                    elif path[0][-1] in ('C1', 'C2', 'C3'):
                        t_states = ['C1', 'C2', 'C3', 'B1', 'H1']
                    else:
                        t_states = ['H1', 'H2', 'H3', 'B1', 'C1']

                    for state in t_states:
                        if transition_probs[last_state].get(state) is not None:
                            if transition_probs[last_state][state][0] != 0 and transition_probs[last_state][state][1] != 0:
                                prior_prob = path[1]
                                trans_prob_r = transition_probs[last_state][state][0]
                                trans_prob_l = transition_probs[last_state][state][1]
                                gauss_prob_r = gaussian_prob(evidence_vector[i][0], emission_paras[state][0])
                                gauss_prob_l = gaussian_prob(evidence_vector[i][1], emission_paras[state][1])
                                current_prob = gauss_prob_r * gauss_prob_l * trans_prob_r * trans_prob_l * prior_prob
                                current_path = path[0][:]
                                current_path.append(state)
                                paths_list.append((current_path, current_prob))
                                values.append(current_prob)
                                '''
                                if current_path == answer[:i+1]:
                                    print(current_path, current_prob, trans_prob_r, trans_prob_l, gauss_prob_r, gauss_prob_l, prior_prob)
                                '''
                                #print(i, current_path, current_prob)
                j = j + 1
            
            if i > 10: 
                p25 = np.percentile(values, 50) 
            
    probability = 0
    sequence = None
    for path in paths_list:

        if len(path[0]) == len(evidence_vector):
            if path[1] > probability:
                probability = path[1]
                sequence = path[0]

    print(probability, sequence)
    return sequence, probability


def return_your_name():
    """Return your name
    """
    name = 'David Jaeyun Kim'
    return name

def MLLR_results_test():

    B_Y  = np.array([[61, 61, 59, 65, 73, 75, 79, 79, 79, 74, 68],
                    [123, 116, 121, 99, 97, 98, 74, 84, 84, 89, 81]])
    '''
    B_Mu = np.array([[1.000, 41.750, 108.200],
                     [1.000, 58.625, 78.670],
                     [1.000, 53.125, 64.182]])
    '''
    B_Mu = np.array([[1.000, 1.000, 1.000],
                     [41.750, 58.625, 53.125],
                     [108.200, 78.670, 64.182]])

    B_X  =  np.array([[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
                     [41.750, 41.750, 41.750, 58.625, 58.625, 58.625, 53.125, 53.125, 53.125, 53.125, 53.125],
                     [108.200, 108.200, 108.200, 78.670, 78.670, 78.670, 64.182, 64.182, 64.182, 64.182, 64.182]])

    B_W = np.dot(np.dot(B_Y, B_X.transpose()), np.linalg.inv(np.dot(B_X, B_X.transpose())))
    B_Mu_N = np.dot(B_W, B_Mu)

    C_Y  = np.array([[44, 53, 62, 64, 66, 59, 61, 58],
                     [73, 70, 78, 62, 58, 51, 76, 90]])
    '''
    C_Mu = np.array([[1.000, 35.667, 56.300],
                     [1.000, 43.667, 37.110],
                     [1.000, 44.200, 50.000]])
    '''
    C_Mu = np.array([[1.000, 1.000, 1.000],
                     [35.667, 43.667, 44.200],
                     [56.300, 37.110, 50.000]])

    C_X  =  np.array([[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
                      [35.667, 35.667, 35.667, 43.667, 43.667, 43.667, 44.200, 44.200],
                      [56.300, 56.300, 56.300, 37.110, 37.110, 37.110, 50.000, 50.000]])

    C_W = np.dot(np.dot(C_Y, C_X.transpose()), np.linalg.inv(np.dot(C_X, C_X.transpose())))
    C_Mu_N = np.dot(C_W, C_Mu)

    H_Y  = np.array([[59, 59, 60, 57, 56, 49, 51, 51, 53, 59, 72, 81, 82, 84, 86, 90],
                     [65, 68, 69, 70, 64, 59, 57, 51, 51, 59, 79, 82, 89, 90, 90, 93]])
    '''
    H_Mu = np.array([[1.000, 45.333, 53.600],
                     [1.000, 34.952, 37.168],
                     [1.000, 67.438, 74.176]])
    '''
    H_Mu = np.array([[1.000, 1.000, 1.000],
                     [45.333, 34.952, 67.438],
                     [53.600, 37.168, 74.176]])

    H_X  =  np.array([[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
                      [45.333, 45.333, 45.333, 45.333, 45.333, 34.952, 34.952, 34.952, 34.952, 34.952, 67.438, 67.438, 67.438, 67.438, 67.438, 67.438],
                      [53.600, 53.600, 53.600, 53.600, 53.600, 37.168, 37.168, 37.168, 37.168, 37.168, 74.176, 74.176, 74.176, 74.176, 74.176, 74.176]])

    H_W = np.dot(np.dot(H_Y, H_X.transpose()), np.linalg.inv(np.dot(H_X, H_X.transpose())))
    H_Mu_N = np.dot(H_W, H_Mu)

    b_emission_paras = {
        'B1': [(B_Mu_N[0][0], 2.773), (B_Mu_N[1][0], 17.314)],
        'B2': [(B_Mu_N[0][1], 5.678), (B_Mu_N[1][1], 1.886)],
        'B3': [(B_Mu_N[0][2], 5.418) ,(B_Mu_N[1][2], 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    c_emission_paras = {
        'C1': [(C_Mu_N[0][0], 4.899), (C_Mu_N[1][0], 10.659)],
        'C2': [(C_Mu_N[0][1], 1.700), (C_Mu_N[1][1], 4.306)],
        'C3': [(C_Mu_N[0][2], 7.341), (C_Mu_N[1][2], 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    h_emission_paras = {
        'H1': [(H_Mu_N[0][0], 3.972), (H_Mu_N[1][0], 7.392)],
        'H2': [(H_Mu_N[0][1], 8.127), (H_Mu_N[1][1], 8.875)],
        'H3': [(H_Mu_N[0][2], 5.733), (H_Mu_N[1][2], 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_emission_paras, 
            c_emission_paras, 
            h_emission_paras)

def MLLR_results_round():

    B_Y  = np.array([[61, 61, 59, 65, 73, 75, 79, 79, 79, 74, 68],
                    [123, 116, 121, 99, 97, 98, 74, 84, 84, 89, 81]])

    B_Mu = np.array([[1.000, 41.750, 108.200],
                     [1.000, 58.625, 78.670],
                     [1.000, 53.125, 64.182]])

    B_X  =  np.array([[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
                    [41.750, 41.750, 41.750, 58.625, 58.625, 58.625, 53.125, 53.125, 53.125, 53.125, 53.125],
                    [108.200, 108.200, 108.200, 78.670, 78.670, 78.670, 64.182, 64.182, 64.182, 64.182, 64.182]])

    B_W = np.dot(np.dot(B_Y, B_X.transpose()), np.linalg.inv((np.dot(B_X, B_X.transpose()))))
    B_Mu_N = np.dot(B_W, B_Mu)

    C_Y  = np.array([[44, 53, 62, 64, 66, 59, 61, 58],
                     [73, 70, 78, 62, 58, 51, 76, 90]])

    C_Mu = np.array([[1.000, 35.667, 56.300],
                     [1.000, 43.667, 37.110],
                     [1.000, 44.200, 50.000]])

    C_X  =  np.array([[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
                      [35.667, 35.667, 35.667, 43.667, 43.667, 43.667, 44.200, 44.200],
                      [56.300, 56.300, 56.300, 37.110, 37.110, 37.110, 50.000, 50.000]])

    C_W = np.dot(np.dot(C_Y, C_X.transpose()), np.linalg.inv((np.dot(C_X, C_X.transpose()))))
    C_Mu_N = np.dot(C_W, C_Mu)

    H_Y  = np.array([[59, 59, 60, 57, 56, 49, 51, 51, 53, 59, 72, 81, 82, 84, 86, 90],
                     [65, 68, 69, 70, 64, 59, 57, 51, 51, 59, 79, 82, 89, 90, 90, 93]])

    H_Mu = np.array([[1.000, 45.333, 53.600],
                     [1.000, 34.952, 37.168],
                     [1.000, 67.438, 74.176]])

    H_X  =  np.array([[1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000],
                      [45.333, 45.333, 45.333, 45.333, 45.333, 34.952, 34.952, 34.952, 34.952, 34.952, 67.438, 67.438, 67.438, 67.438, 67.438, 67.438],
                      [53.600, 56.300, 56.300, 53.600, 53.600, 37.168, 37.168, 37.168, 37.168, 37.168, 74.176, 74.176, 74.176, 74.176, 74.176, 74.176]])

    H_W = np.dot(np.dot(C_Y, C_X.transpose()), np.linalg.inv((np.dot(C_X, C_X.transpose()))))
    H_Mu_N = np.dot(C_W, C_Mu)

    b_emission_paras = {
        'B1': [(round(B_Mu_N[0][0],3), 2.773), (round(B_Mu_N[1][0],3), 17.314)],
        'B2': [(round(B_Mu_N[0][1],3), 5.678), (round(B_Mu_N[1][1],3), 1.886)],
        'B3': [(round(B_Mu_N[0][2],3), 5.418) ,(round(B_Mu_N[1][2],3), 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    c_emission_paras = {
        'C1': [(round(C_Mu_N[0][0],3), 4.899), (round(C_Mu_N[1][0],3), 10.659)],
        'C2': [(round(C_Mu_N[0][1],3), 1.700), (round(C_Mu_N[1][1],3), 4.306)],
        'C3': [(round(C_Mu_N[0][2],3), 7.341), (round(C_Mu_N[1][2],3), 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    h_emission_paras = {
        'H1': [(round(H_Mu_N[0][0],3), 3.972), (round(H_Mu_N[1][0],3), 7.392)],
        'H2': [(round(H_Mu_N[0][1],3), 8.127), (round(H_Mu_N[1][1],3), 8.875)],
        'H3': [(round(H_Mu_N[0][2],3), 5.733), (round(H_Mu_N[1][2],3), 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_emission_paras, 
            c_emission_paras, 
            h_emission_paras)

def MLLR_results():

    b_emission_paras = {
        'B1': [(60.333, 2.773), (120.000, 17.314)],
        'B2': [(71.000, 5.678), (98.000, 1.886)],
        'B3': [(75.800, 5.418), (82.400, 5.573)],
        'Bend': [(None, None), (None, None)]
    }

    c_emission_paras = {
        'C1': [(53.000, 4.899), (73.667, 10.659)],
        'C2': [(63.000, 1.700), (57.000, 4.306)],
        'C3': [(59.500, 7.341), (83.000, 7.826)],
        'Cend': [(None, None), (None, None)]
    }

    h_emission_paras = {
        'H1': [(58.200, 3.972), (67.200, 7.392)],
        'H2': [(52.600, 8.127), (55.400, 8.875)],
        'H3': [(82.500, 5.733), (87.167, 8.347)],
        'Hend': [(None, None), (None, None)]
    }

    return (b_emission_paras, 
            c_emission_paras, 
            h_emission_paras)