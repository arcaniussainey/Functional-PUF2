#Importing various libraries. use "pip install {lib_name}" to install the required libraries.
import jax.numpy as jnp
from Pufs.FunctionalPuf import *
from pypuf import *
import time
from typing import Dict
import csv
from evosax import Strategies # type: ignore
import jax

# Initializing an empty list called LOG.
LOG = []

#Function to log the data and append it to LOG list.
def log_data(data: Dict, log_path="Funlog/log.csv"):
    global LOG
    LOG.append(data)

#Function to convert array like object to strings - join them - calculate hash result of the string.
def hash_w(w):
    w_str = "".join([str(x) for x in w.flatten().tolist()])
    return hash(w_str)


# @partial(jax.jit, static_argnums=(1, 2, 3)) . jit is just in time compilation.
def run_es_loop(rng, num_steps, fit_fn, model):
    """

    scan through evolution rollouts

    tldr: your function should take (rng, w) as its first two parameters parameters
    and should not read or write from external state.

    this is is necessary to ensure the inner loop of the training is as efficient as possible.

    see https://jax.readthedocs.io/en/latest/jax-101/07-state.html

    see: "Scan Through Evolution Rollouts" (https://github.com/RobertTLange/evosax)

    Args:
        rng (PRNG): a prng key ---- "Pseudo" random number generators (PRNG).
        n_generations (int): number of generations

    Returns:
        tuple(jnp.array, Any): (mean, state)
    """
    es_params = model.default_params
    state = model.initialize(rng, es_params)

    #Inside the function, it initializes the model and then iterates through evolution steps, 
    #repeatedly asking the model for solutions and updating its state based on fitness evaluations.

    def es_step(state_input, tmp):
        #state_input is a tuple containing the current random number generator key (rng) and the state of the ES optimization process.
        rng, state = state_input    

        # The purpose of splitting the key is to ensure that RNG remains reproducible .
        rng, rng_iter = jax.random.split(rng)

        #The ask function is typically responsible for generating a set of candidate solutions (x) based on the current state and parameters of the ES algorithm.
        x, state = model.ask(rng_iter, state, es_params) 
        #Calculates the fitness values for the candidate solutions (x) using the fitness function (fit_fn). 
        fitness = fit_fn(rng_iter, x).flatten() 
        # The tell function is responsible for updating the ES state based on the candidate solutions (x) and their fitness values. 
        state = model.tell(x, fitness, state, es_params)
        #returns list of PRNG keys, updates state and min fitness. This minimum fitness value is often used as an indicator of the progress made in the optimization process
        return [rng, state], fitness[jnp.argmin(fitness)] 

    #jax.lax.scan function perform multiple iterations of the es_step function and collect the results. 
    state, scan_out = jax.lax.scan(es_step, [rng, state], [jnp.zeros(num_steps)]) 
    return jnp.mean(scan_out), state  #Returns mean of all fitness values collected in scan_out


#This function is similar to previous run_es_loop function but it adds noise into the process.
def noisy_run_es_loop(rng, num_steps, fit_fn, model, sigma_error):  
    es_params = model.default_params
    state = model.initialize(rng, es_params)
   
    def es_step(state_input, tmp):
        rng, state = state_input
        #Splits PRNG key  into four subkeys 
        rng, *subkey = jax.random.split(rng, 4)   
        x, state = model.ask(subkey[0], state, es_params)

        #The function appears to introduce noise into the candidate solutions to create a set of perturbed solutions. 
        rng, noisy_weight = noisy_generate_weights(subkey[1], x, sigma_error) 
        fitness = fit_fn(subkey[2], noisy_weight) #It evaluates the fitness of the noisy weights
        state = model.tell(x, fitness.flatten(), state, es_params)
        return [rng, state], fitness[jnp.argmin(fitness)]


    state, scan_out = jax.lax.scan(es_step, [rng, state], [jnp.zeros(num_steps)])
    return jnp.mean(scan_out), state


# returns two arrays, one representing the sorted arr array and the other representing the sorted on values. 
@jax.jit
def sort_on(arr, on):

    """
    Sorts the array <arr> based on the values in array <on>.

    Args:
        arr (numpy.ndarray): Array to be sorted.
        on (numpy.ndarray): Array used for sorting.

    Returns:
        tuple: A tuple containing two arrays - the first array represents the sorted arr array,
               and the second array represents the values in array on sorted accordingly.
    """

    # sortby <on> reshuffle arr to match
    sorted_on, sort_idxs = jax.lax.sort_key_val(on, jnp.arange(0, on.shape[0])) #sorted_on is the sorted version of the on array, 
                                                                                #sort_idxs: indices that specify the order in which 'on' elements have been rearranged.
    sorted_arr = arr[sort_idxs] # it sorts arr based on the values in the on array.(matching the sorted order)
    return sorted_arr, sorted_on



def fitness_direct_comparison(rng, w, w2, c):

    """
    Computes the fitness value based on direct comparison of responses from two weight arrays.

    Args:
        rng (numpy.ndarray): Random number generator.
        w (numpy.ndarray): Weight array for the first configuration.
        w2 (numpy.ndarray): Weight array for the second configuration.
        c (numpy.ndarray): Matrix of challenges.

    Returns:
        float: Mean fitness value computed from the bitwise XOR operation on response arrays.
    """
    #get_response function computes a response or output based on the given parameters and configuration.
    r1 = get_response(w, c).flatten()     
    r2 = get_response(w2, c).flatten()

    #The fitness value is calculated by performing a bitwise XOR operation 
    fitness = jnp.bitwise_xor(r1, r2)  
    return fitness.mean()



def filter_challenges(puf, c, threshold):

    """
    Filters a matrix of challenges based on acceptance criteria derived from delta responses.

    Args:
        puf: Physical Unclonable Function object.
        c (array-like): Matrix of challenges.
        threshold (float): Threshold for acceptance criteria.

    Returns:
        numpy.ndarray: Matrix containing only the challenges that meet the acceptance criteria.
    """
    # calculates delta response, which is the raw time delay value. Resulting delta is a matrix where each row represents the delta response corresponding to a specific challenge.
    delta = get_delta_response(puf.weight, c)  
    #sigma values represent a measure of the variation in responses for each challenge.                                            
    sigma = delta.std(axis=0)  

    #Each element of the accept array indicates whether the delta responses for a challenge meet the acceptance criteria based on the threshold.
    accept = jax.vmap( 
        lambda d: jnp.greater_equal(jnp.abs(d), sigma * threshold).sum(), in_axes=(0,) #The jax.vmap function applies lambda function to each row of delta and returns the results as an array.
    )#lambda function, it computes the absolute value of d, compares it to a threshold calculated as sigma * threshold, and sums up the results.
    
    # Both are compared to see if all challenges meet the acceptance criteria.
    accept_idx = jnp.equal(accept(delta), delta.shape[1]) 
    # The function filters the original challenge matrix c based on acceptance criteria.
    filt_chall = c[accept_idx, :] 
    # Returns resulting matrix contains only the challenges that meet the acceptance criteria.
    return filt_chall 


def fitness_chall_filter(rng, w, c, threshold):

    """
    Computes the fitness value based on filtered challenges and a given threshold.
    It checks whether challenges meet specific criteria, and the fitness score represents the proportion of challenges that meet
    these criteria.

    Args:
        rng (numpy.ndarray): Random number generator.
        w (numpy.ndarray): Weight array.
        c (numpy.ndarray): Matrix of challenges.
        threshold (float): Threshold for acceptance criteria.

    Returns:
        float: Fitness value calculated based on filtered challenges.
    """

    rng, subkey = jax.random.split(rng) #Splitting keys

    #Estimate of the standard deviation of delta responses.
    sigma_hat = get_delta_response(w, c_sample).std(axis=0).reshape((1, -1)) 
    #Calculates delta responses (delta) for the original set of challenges c.
    delta = get_delta_response(w, c) 

    #Each element of the accept array indicates whether the delta responses for a challenge meet the acceptance criteria based on the threshold.
    accept = jax.vmap(
        lambda _delta: jnp.greater_equal(jnp.abs(_delta), sigma_hat * threshold).sum(),
        in_axes=(0,),
    )
    #This comparison effectively checks if there are any challenges that do not meet the specified criteria and calculates fitness.
    fitness = jnp.not_equal(accept(delta), w.shape[0]) 
    fitness = fitness.mean().astype(jnp.float32) #mean of fitness value is calculated.
    return fitness


def xorn_fitness(rng, w, prev_ws, valid_chall, threshold, alpha):                   #alpha: This is a penalty coefficient.
    """
    Calculates a fitness score for a given weight vector w based on challenges, comparisons with previous weights,
    and a penalty term. The penalty term penalizes weights that do not perform well compared to previous weights. 
    The final fitness score is a combination of the filtered fitness score and the scaled penalty.

    Args:
        rng (numpy.ndarray): Random number generator.
        w (numpy.ndarray): Weight vector.
        prev_ws (List[numpy.ndarray]): List of previous weight vectors.
        valid_chall (numpy.ndarray): Matrix of valid challenges.
        threshold (float): Threshold for filtering challenges.
        alpha (float): Penalty coefficient.

    Returns:
        float: Final fitness score.
    """
    
    filter_score = fitness_chall_filter(rng, row_vec(w), valid_chall, threshold)   
    prev_eql = jax.vmap(
        lambda _prev_weight: fitness_direct_comparison(    
            rng, row_vec(w), row_vec(_prev_weight), valid_chall 
        )
    )
    #This results in an array that represents the percentage of times that w is equal in fitness to each previous weight.
    pct_eql = prev_eql(prev_ws)  
    #penalty calculated based on whether pct_eql is greater than 1 - pct_eql and assigns either pct_eql or 1 - pct_eql.  
    penalty = (jnp.where(pct_eql > 1 - pct_eql, pct_eql, 1 - pct_eql) * 2).mean() 
    
    #scaling the penalty
    penalty = penalty * alpha   
    #final fitness score is avg of filter score and scaled penalty.
    fitness = (filter_score + penalty) / 2 
    return fitness


def arbiter_direct_attack(chall, pop_size=32, n_generations=500, model_name="CMA_ES"):

    """
    Performs an evolutionary optimization attack on an Arbiter PUF using the specified algorithm.

    Args:
        chall (numpy.ndarray): Matrix of challenges.
        pop_size (int): Population size for the optimization algorithm.
        n_generations (int): Number of generations for the optimization algorithm.
        model_name (str): Name of the optimization algorithm (e.g., "CMA_ES").

    Returns:
        tuple: A tuple containing the final state of the optimization and the learned weight vector.
    """

    nchall, dim = chall.shape  
    #Arbiter PUF is initialized with a new random key and shape (1,dim)
    arb = Arbiter(new_key(), (1, dim)) 

    #fitness_direct_comparison function compares the given weight w with the weight of the Arbiter PUF (arb.weight) for each challenge in chall.
    arb_direct_fitness = jax.vmap(
        lambda rng, w: fitness_direct_comparison(rng, row_vec(w), arb.weight, chall),
        in_axes=(None, 0),
    )

    #Compiled for improved performance using just in time compilation 
    arb_direct_fitness = jax.jit(arb_direct_fitness) 

    #initializing optimization stratergy model.
    model = Strategies[model_name](popsize=pop_size, num_dims=dim) 

    #The default parameters for the evolutionary strategy (ES) model are obtained.
    es_params = model.default_params 
    
    #The ES model is initialized with a new random key and the default ES parameters.
    state = model.initialize(new_key(), es_params) 
    #This optimization loop aims to find a weight vector that maximizes the fitness according to the arb_direct_fitness function.
    _, state = run_es_loop(new_key(), n_generations, arb_direct_fitness, model) 
    _, state = state #Extracting Best Member

    #Learned weight vector is obtained by reshaping the best member from the optimization into a row vector using row_vec.
    lrn_w = row_vec(state.best_member) 
    return state, lrn_w


def x1_attk(
    chall,
    pop_size=32,
    n_generations=500,
    model_name="CMA_ES",
    threshold=2,
    sigma_error=None,
):
    
    """
    Conducts an evolutionary optimization attack on a system using a set of challenges.
    It aims to find the best weight vector that maximizes a fitness function based on the responses to these challenges.

    Args:
        chall (numpy.ndarray): Matrix of challenges.
        pop_size (int): Population size for the optimization algorithm.
        n_generations (int): Number of generations for the optimization algorithm.
        model_name (str): Name of the optimization algorithm (e.g., "CMA_ES").
        threshold (float): Threshold for filtering challenges.
        sigma_error (float or None): Standard deviation of the noise for noisy optimization (if applicable).

    Returns:
        tuple: A tuple containing the final state of the optimization and the learned weight vector.
    """
    #Calculates no.of challenges and the dimension of each challenge based on the shape of the chall matrix
    _, dim = chall.shape  

    #Initializing optimization stratergy model.
    model = Strategies[model_name](popsize=pop_size, num_dims=dim) 
    x1_fit = jax.vmap(
        # w is the candidate in es_step whose fitness needs to be calculated.It evaluates the fitness of a weight vector w with respect to the challenges in chall using the specified threshold.
        lambda rng, w: fitness_chall_filter(rng, row_vec(w), chall, threshold), 
     
        in_axes=(None, 0),
    )

    '''This conditional block checks if sigma_error is specified it runs an evolutionary optimization loop with noise (noisy_run_es_loop). 
    using a new random key (new_key()), the specified number of generations (n_generations), the fitness function x1_fit,
    the optimization model model, and the provided sigma_error.
    '''
    if sigma_error != None:
        _, state = noisy_run_es_loop(
            new_key(), n_generations, x1_fit, model, sigma_error
        )

    else:
        _, state = run_es_loop(new_key(), n_generations, x1_fit, model) #If sigma_error is not specified (None),
        #It runs a standard evolutionary optimization loop (run_es_loop) using the same parameters but without noise.

    #Extracting Best Member
    _, state = state 
    # The best member's weight vector from the optimization is extracted and reshaped into a row vector 
    lrn_w = row_vec(state.best_member) 
    #Learned weight vector, stacked vertically as a matrix
    return state, jnp.vstack(lrn_w)  

'''
Note:
x1_attk evaluates the fitness of a weight vector using a single set of challenges (chall).
xn_attk evaluates the fitness of a weight vector using multiple sets of challenges (chall) and takes into account previous 
weights (prev_w) in the fitness calculation.
xn_attk is more complex and versatile, allowing you to conduct optimization attacks that consider both multiple sets of 
challenges and the impact of previous weights on the fitness of the current weight vector. It's suitable for scenarios
where you have multiple datasets or sets of challenges and want to optimize the weight vector accordingly.'''

def xn_attk(
    chall,
    prev_w,
    pop_size=32,
    n_generations=500,
    model_name="CMA_ES",
    threshold=2,
    alpha=0.43,   # alpha coeff and previous weights
    sigma_error=None,
):
    """
    Conducts an evolutionary optimization attack on a system using multiple sets of challenges and previous weights.

    Args:
        chall (numpy.ndarray): Matrix of challenges.
        prev_w (List[numpy.ndarray]): List of previous weight vectors.
        pop_size (int): Population size for the optimization algorithm.
        n_generations (int): Number of generations for the optimization algorithm.
        model_name (str): Name of the optimization algorithm (e.g., "CMA_ES").
        threshold (float): Threshold for filtering challenges.
        alpha (float): Coefficient for penalty term and previous weights.
        sigma_error (float or None): Standard deviation of the noise for noisy optimization (if applicable).

    Returns:
        tuple: A tuple containing the final state of the optimization and the learned weight vector.
    """
    
    _, dim = chall.shape
    model = Strategies[model_name](popsize=pop_size, num_dims=dim)

    def xn_fit(rng, w):
        return xorn_fitness(rng, row_vec(w), prev_w, chall, threshold, alpha) 
                                                                            
    #vectorizes the xn_fit function to handle multiple weight vectors efficiently
    xn_fit = jax.vmap(xn_fit, in_axes=(None, 0)) 

    if sigma_error != None:
        _, state = noisy_run_es_loop(
            new_key(), n_generations, xn_fit, model, sigma_error
        )
    else:
        _, state = run_es_loop(new_key(), n_generations, xn_fit, model)
    _, state = state
    lrn_w = row_vec(state.best_member)
    return state, lrn_w #Explanation below.

'''The returned EvoState (state) Object:

p_sigma: This is an array of floats representing the evolution of the step size for each dimension during the optimization process.
p_c: Similar to p_sigma, this is an array of floats representing the evolution of the covariance matrix adaptation.
C: This is a covariance matrix, likely related to the covariance matrix adaptation algorithm used in the optimization process.
D: This seems to be a variable that is not specified or provided in the output.
B: Similar to D, it also appears to be a variable not specified or provided in the output.
mean: An array of floats representing the mean of the population during the optimization process.
sigma: A single float representing the step size or standard deviation.
weights: An array of floats representing the weights associated with each dimension during optimization.
weights_truncated: Similar to weights, but it seems to be truncated after a certain point.
best_member: An array of floats representing the best member or solution found during the optimization process.  the best_member is a critical part of the optimization output as it represents the optimal solution found by the algorithm.
best_fitness: A single float representing the fitness or objective value of the best member found.
gen_counter: An integer representing the number of generations or iterations performed during the optimization process.
'''



def xor_cfilt_data(n_trials):

    """
    Conducts a wide range of experiments by varying challenge dimensions, the number of challenges, filtering thresholds, and the number of XOR operations.
    Records and stores the results in a CSV file.

    Args:
        n_trials (int): Number of trials to run for each experiment configuration.

    Returns:
        list: A list of dictionaries containing the experimental results.
    """

    dim = [64, 128]
    nchall = [10_000, 20_000, 50_000, 100_000]
    thresholds = [1, 1.5, 2, 2.5, 3]
    num_xor = [2, 3, 4, 5, 6]   
    data = [] 
    #These loops generate a wide range of experiments.
    for threshold in thresholds:    
        for d in dim:
            for n in nchall:
                for nxor in num_xor:
                    for _ in range(n_trials):
                        c = generate_challenges(new_key(), (n, d))   
                        xor = Xor(new_key(), (nxor, d)) 
                        #Challenges are filtered using the filter_challenges
                        fc = filter_challenges(xor, c, threshold) 
                        #This variable stores the number of challenges that pass the filtering criteria
                        num_kept = fc.shape[0] 
                        #It calculates the percentage of challenges that are kept after filtering.
                        pct_kept = num_kept / n 
                        data.append(
                            dict(
                                nchall=n,
                                dim=d,
                                threshold=threshold,
                                num_kept=num_kept,
                                pct_kept=pct_kept,
                                nxor=nxor,
                            ) #For each experiment run, a dictionary is created with experiment details
                        )

    with open("data/xor_challenge_filter_data.csv", "a+") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()

        #Collected data is stored in csv.
        writer.writerows(data) 

    return data


def similarity(w, compar_w):

    """
    Calculates a similarity metric between a given weight vector and a set of comparison weight vectors.
    Assesses how similar the behavior of w is to the behavior of other weight vectors in compar_w.

    Args:
        w (numpy.ndarray): The weight vector for comparison.
        compar_w (numpy.ndarray): Set of comparison weight vectors.

    Returns:
        numpy.ndarray: Array containing the similarity values between the given weight vector and each weight vector in the comparison set.
    """

    c = generate_challenges(new_key(), (25_000, w.shape[1])) #The challenges are generated based on a new_key() and the shape of w. 25_000 challenges are generated.
    sim = jax.vmap(
        #Calculate the similarity between w and each weight vector w2 in compar_w. The similarity is calculated as the mean of a Boolean array that checks whether the responses of w and w2 to the challenges are equal.
        lambda w2: jnp.equal(      
            #Calculates the responses of the weight vector w to the challenges c 
            get_response(row_vec(w), c).flatten(), 
            get_response(row_vec(w2), c).flatten(),
        ).mean()
    )
    #Returns the similarity values between w and each weight vector in compar_w and returns these values as an array.
    return sim(compar_w) 


def initialize_challenge_filter(nchall, puf, threshold):

    """
    Generates an initial set of challenges, applies a threshold-based filter to them, and then iteratively generates more
    challenges while gradually increasing the factor until the desired number of challenges is obtained. 

    Args:
        nchall (int): Desired number of challenges.
        puf (PUF): Physical Unclonable Function object.
        threshold (float): Threshold value for filtering challenges.

    Returns:
        numpy.ndarray: Matrix containing the filtered challenges.
    """

    # Generate an initial set of challenges 'chall'
    chall = generate_challenges(new_key(), (nchall, (puf.dim[1])))
    
    # Filter the initial challenges based on the PUF and threshold
    filt_chall = filter_challenges(puf, chall, threshold)

    
    # Initialize a factor for increasing the number of challenges
    factor = 2
    
    # Continue generating and filtering challenges until reaching the desired number 'nchall'
    while filt_chall.shape[0] < nchall:
        # Increase the factor for generating more challenges
        factor = factor * 2
        
        # Generate a new set of challenges based on the increased factor
        chall = generate_challenges(new_key(), (int(nchall * factor), (puf.dim[1])))
        
        # Filter the newly generated challenges
        filt_chall = filter_challenges(puf, chall, threshold)
    
    # Return the first 'nchall' challenges from the filtered set
    
    return filt_chall[:nchall, :]



def attack_indv_puf(challenges,attk_args,noise,xor,nxor,alphas):
    """
    Conducts an attack on a PUF using a set of challenges and specified attack parameters.

    Args:
        challenges (numpy.ndarray): Set of challenges used in the attack.
        attk_args (dict): Dictionary containing attack parameters.
        noise (bool): Flag indicating whether noise is enabled or not.
        xor (PUF): XOR PUF object.
        nxor (int): Number of XOR operations.
        alphas (list): List of alpha coefficients for subsequent attacks.

    Returns:
        tuple: A tuple containing learned weights, best fitness values, and total execution time.
    """
    
    best_fitness=[]

    # Record the start time for measuring execution time
    start_time = time.monotonic()

    # Run the first attack (x1_attk) with or without noise
    if noise:
        sigma_error = get_sigma_error(new_key(), xor.weight) #Line 428 FuncPuf.
        state1, lrn_w1 = x1_attk(
            challenges, **attk_args, sigma_error=sigma_error[0] #Function at line 191
        )
    else:
        state1, lrn_w1 = x1_attk(challenges, **attk_args) #Function at line 191

    best_fitness.append(state1.best_fitness)

    # Initialize the learned weights
    lrn_w = row_vec(lrn_w1)

    # Store the state of the attack
    states = [state1]

    # Initialize the sigma index for noise if enabled
    sigma_idx = 1

    alpha_len=nxor-1

    # Loop over alpha values for subsequent attacks
    for alpha in alphas[0:alpha_len]:   # Added the condition to take alpha values, 1 less than the number of xors
        attk_args["alpha"] = alpha

        # Run subsequent attacks (xn_attk) with or without noise
        if noise:
            state, lw = xn_attk(
                challenges, lrn_w, **attk_args, sigma_error=sigma_error[sigma_idx] #Function at line 231
            )
            sigma_idx += 1
        else:
        
            state, lw = xn_attk(challenges, lrn_w, **attk_args) #Function at line 231

        best_fitness.append(state.best_fitness)

        # Store the state of the current attack
        states.append(state)

        # Update the learned weights
        lrn_w = jnp.vstack([lrn_w, row_vec(lw)])


    # Record the end time for measuring execution time
    end_time = time.monotonic()

    total_time = end_time-start_time

    return lrn_w,best_fitness,total_time


def transform_flip(C1,R1):

    """
    Flips the bits of challenges based on the responses received from the Interpose PUF.

    Args:
        C1 (numpy.ndarray): Original challenges.
        R1 (numpy.ndarray): Responses received from a iPUF.

    Returns:
        numpy.ndarray: Transformed challenges with flipped bits.
    """

    R1 = jnp.where(jnp.array(R1) == 1, 1, -1)

    # Create a list to store the new challenges
    C2 =[]  
    mid_index = C1.shape[1] // 2

    for i in range(len(C1)):
        # Check if the element in R1 matches the element in C1 at the mid_index
        if R1[i] == -1:
            # If the condition is true, insert the element from R1 at mid_index in C1[i]
            C1_i = jnp.insert(C1[i], mid_index, R1[i])
            C2.append(C1_i)
        else:
            # If the condition is false, modify the middle element in C1[i] by adding R1[i]
            C1_i = jnp.insert(C1[i], mid_index, R1[i])  # Update the middle element
            # Flip elements before mid_index using array multiplication
            C1_i_flipped = -1 * C1_i[:mid_index]

            # Concatenate the flipped elements with the remaining elements
            C_transformed = jnp.concatenate((C1_i_flipped, C1_i[mid_index:]))
            C2.append(C_transformed)

    transformed_challenges=jnp.array(C2)
    return transformed_challenges


def reverse_transform(C2):

    """
    Reverse transforms challenges to get back C1 (Challenges before transformation, 
    which is filtered as per the right XOR-PUF.

    Args:
        C2 (numpy.ndarray): Transformed challenges.

    Returns:
        numpy.ndarray: Reverse transformed challenges.
    """

    # Calculate the index of the middle element
    mid_index = len(C2[0]) // 2
    C_filtered_list = []

    for i in range(len(C2)):

        if C2[i][mid_index] == 1:

            # If the middle element is 1, flip the bits before the middle index
            flip = -1 * C2[i][:mid_index]
            modified_challenge = jnp.concatenate((flip, C2[i][mid_index+1:]))

        else:
             # If the middle element is -1, keep the bits before and after the middle index unchanged
            modified_challenge = jnp.concatenate([C2[i][:mid_index], C2[i][mid_index + 1:]])

        C_filtered_list.append(modified_challenge)

    C_filtered_list=jnp.array(C_filtered_list)

    return C_filtered_list


def initialize_ipuf_challenge_filter(nchall, lxor,rxor, threshold): 

    """
    Initializes challenges C1, C2, and C_filtered for an Interpose PUF (IPUF).

    Args:
        nchall (int): Number of challenges to generate.
        lxor (Xor): Left XOR PUF weights.
        rxor (Xor): Right XOR PUF weights.
        threshold (float): Threshold for challenge filtering.

    Returns:
        numpy.ndarray: Filtered challenges C_filtered.
        numpy.ndarray: Initial challenges C1.
        numpy.ndarray: Responses for C2.
    """

    # Empty list of challenges C2   
    C2=[]

    # Initialize a factor for increasing the number of challenges
    factor = 500
    counter=0
    while len(C2) < nchall:
        # Increase the factor for generating more challenges
        C2=list(C2)
        factor = factor * 100
        
        # Generate a new set of challenges based on the increased factor
        chall = generate_challenges(new_key(), (int(nchall * factor), (lxor.dim[1])))
        
        # Filter the newly generated challenges using the left XOR PUF
        C1 = filter_challenges(lxor, chall, threshold)    
        
        # Calculate true responses by passing C1 to the left XOR PUF
        # individual shape is (N, k) -- only the XOR result is used here
        true_r_individual,true_r = lxor(C1)
        R1=true_r.flatten()
        
        # Transform the challenges based on responses to obtain C2
        C2=transform_flip(C1,R1)   

        # Filter the challenges C2 using the right XOR PUF
        C2=filter_challenges(rxor,C2,threshold)

    # Take the first 'nchall' challenges from C2
    C2=C2[:nchall, :]

    # Get the responses for C2 from the right XOR PUF
    C2_R_indv,C2_R= rxor(C2)

    # Reverse transform to obtain C1 from C2 which is filtered as per both left and right XOR PUF.
    C_filtered=reverse_transform(C2)

    C_filtered=C_filtered[:nchall, :]

    # Return the first 'nchall' challenges of both C_filtered and C2_R. 
    return C_filtered,C2,C2_R

'''
Function performs multiple attack simulations on an XOR-based PUF, records the results, and returns valuable information about the 
attack's effectiveness under different scenarios and configurations. The function is designed for thorough analysis and evaluation 
of PUF security.
'''
def xor_attk(
    dim=(3, 128), 
    nchall=10000, 
    threshold=0.5, 
    n_generations=1000, 
    model_name="CMA_ES", 
    pop_size=32,  
    alphas=[0.37,0.55,0.16,0.39,0.12,0.23,0.31,0.47,0.29, 0.34, 0.18, 0.26], 
    noise=False,
    rxor=0,
):
    
    """
    Performs multiple attack simulations on an XOR-based PUF.

    Args:
        dim (tuple): Specifies the number of XOR gates and the dimension of each XOR gate. Default is (3, 128).
        nchall (int): Number of challenge-response pairs (CRPs) to be generated and used in the attack. Default is 10000.
        threshold (float): Threshold used for filtering challenges. Default is 0.5.
        n_generations (int): Number of iterations to be performed. Default is 1000.
        model_name (str): Name of the optimization algorithm/model to be used. Default is "CMA_ES".
        pop_size (int): Number of potential solutions evaluated in each generation of the algorithm. Default is 32.
        alphas (list): Tuning parameters for the optimization process. Default is [0.37, 0.55, 0.16, 0.39, 0.12, 0.23, 0.31, 0.47, 0.29, 0.34, 0.18, 0.26].
        noise (bool): Flag to enable noise in the attack. Default is False.
        rxor (int): Flag to indicate the use of an IPUF (default is 0 for XOR PUF).

    Returns:
        dict: Dictionary containing collected data from the attack simulations.
    """

    #Extract the number of XOR gates (nxor) and the dimension (dim) from the input dim tuple.
    nxor, dim = dim

    accuracies = []

   # Create an XOR PUF instance with specified parameters
    xor = Xor(new_key(), (nxor, dim))
    print("The original weights",xor)
    
     # Initialize challenge filter based on the PUF type
    if rxor ==0 :
        xor_filt_chall1 = initialize_challenge_filter(nchall, xor, threshold) #Function at line 316.
        C_filtered=xor_filt_chall1
    else:
        C_filtered,C2,C2_R = initialize_ipuf_challenge_filter(nchall, xor,rxor, threshold)
    best_fitness=[]

    # Generate a sample set of challenges for filtering
    global c_sample
    c_sample = generate_challenges(new_key(), (5000, C_filtered.shape[1]))

    # Generate a validation set of challenges and Filter challenges using the PUF
    val_chall = filter_challenges(                                      
        xor, generate_challenges(new_key(), (25_000, (dim))), threshold   
    )

    # Initialize variables for accuracy calculation
    overall_acc=0.45
    while overall_acc<0.98:

        # Define attack arguments
        attk_args = dict(
        pop_size=pop_size,
        n_generations=n_generations,
        model_name=model_name,
        threshold=threshold,
    )
        # Perform PUF attack
        lrn_w,best_fitness,total_time=attack_indv_puf(C_filtered,attk_args,noise,xor,nxor,alphas)

        # Calculate accuracy of learned PUF on validation challenges
        # individual shape is now (N, k) -- access arbiter j as individual[:, j]
        val_old_xor_r_individual, val_old_xor_r=xor(val_chall)
        val_old_xor_r=val_old_xor_r.flatten()

        #Compute the learned responses of the XOR PUF to the validation challenges.
        lrn_r_individual,lrn_r = xor_get_response(lrn_w, val_chall) 
        lrn_r=lrn_r.flatten()

        acc = jnp.equal(val_old_xor_r, lrn_r).mean()
        overall_acc = jnp.array([acc, 1 - acc]).max()
        
    #This is being done to use actual challenges to use in right xor in ipuf:
    lrn_r_individual_actual_chal,lrn_r_actual_chal = xor_get_response(lrn_w, C_filtered) #Function at line 372 FuncPuf
    lrn_r_actual_chal=lrn_r_actual_chal.flatten()

    # Calculate the accuracy of the learned PUF on validation challenges
    acc = jnp.equal(val_old_xor_r, lrn_r).mean() # we use the first xor instance and give it validation challenges and compare with new xor response.
    print("Accuracy",acc)
    overall_acc = jnp.array([acc, 1 - acc]).max()
    accuracies.append(overall_acc)

    # Calculate and store individual accuracies
    indv_acc=[]

    # Iterate over the XOR instances.
    # individual is (N, k): column j holds all of arbiter j's responses.
    for i in range(nxor):
        lrn_r_subset = lrn_r_individual[:, i]

        # Iterate over all arbiter columns of the true individual responses
        for j in range(nxor):
            r_subset = val_old_xor_r_individual[:, j]

            # Calculate and print the accuracy
            this_acc = jnp.equal(r_subset, lrn_r_subset).mean()
            print(f"Prediction acc wv_{i + 1} vs wv_{j + 1}", this_acc)
            indv_acc.append(this_acc)


    # Print and return results and data
    print(f"nchall={nchall}: {overall_acc.mean()}")

    # Calculate final fitness
    final_fitness = fitness_chall_filter(   
        new_key(), lrn_w, C_filtered, threshold
    ).item()

    if(rxor==0):
        C2_R,C2=0,0
    # Create a dictionary to store collected data
    data = dict(
        overall_acc=overall_acc,
        model_name=model_name,
        dim=dim,
        pop_size=pop_size,
        n_generations=n_generations,
        threshold=threshold,
        nchall=nchall,
        alphas=alphas,
        final_fitness=final_fitness,
        noise=noise,
        puf_type=f"xor {xor.dim}",
        weight_hash=hash_w(xor.weight),
        nxors=nxor,
        best_fitness=best_fitness,
        total_time=total_time,
        indv_acc=indv_acc,
        xor_filt_chall=C_filtered,
        lrn_r=lrn_r_actual_chal,
        C2_R=C2_R,
        C2=C2

    )

    # Add alpha values to the data dictionary
    for i in range(len(alphas)):
        data[f"alpha{i+1}"] = alphas[i]

    # Fill remaining alpha slots in the data dictionary with None
    for i in range(len(alphas), 5):
        data[f"alpha{i+1}"] = None

    # Return XOR PUF, learned XOR PUF, attack states, and collected data
    print("TOTAL TIME IS",total_time)
    print(data)
    return data



'''This is the code for the new Ipuf attack '''
def ipuf_attk(
    xor1_dim=(3,128),
    dim=(3, 129),
    nchall=5000, 
    threshold=0.75, 
    n_generations=1000, 
    model_name="CMA_ES", 
    pop_size=32,  
    alphas=[0.37,0.55,0.16,0.39,0.12,0.23,0.31,0.47,0.29, 0.34, 0.18, 0.26], 
    noise=False, 
):
    
    """
    Performs an attack on an InterposePUF (IPUF) configuration.

    Args:
        xor1_dim (tuple): Dimensions of the first XOR PUF. Default is (3, 128).
        dim (tuple): Dimensions of the IPUF (always one stage higher than the left XOR). Default is (3, 129).
        nchall (int): Number of challenge-response pairs (CRPs) to be generated and used in the attack. Default is 5000.
        threshold (float): Threshold used for filtering challenges. Default is 0.75.
        n_generations (int): Number of iterations to be performed. Default is 1000.
        model_name (str): Name of the optimization algorithm/model to be used. Default is "CMA_ES".
        pop_size (int): Number of potential solutions evaluated in each generation of the algorithm. Default is 32.
        alphas (list): Tuning parameters for the optimization process. Default is [0.37, 0.55, 0.16, 0.39, 0.12, 0.23, 0.31, 0.47, 0.29, 0.34, 0.18, 0.26].
        noise (bool): Flag to enable noise in the attack. Default is False.

    Returns:
        dict: Dictionary containing collected data from the attack simulations.
    """

    #Extract the number of XOR gates (nxor) and the dimension (dim) from the input dim tuple.
    nxor, dim = dim

    # List to store accuracy values
    accuracies = []

    # Create an XOR PUF instance with nxor PUFs, each having dim dimensions. Initialize it with a new random key.
    xor = Xor(new_key(), (nxor, dim))
    print("The original weights of iPUF",xor)

    # Retrieve data from XOR PUF attack(Left XOR)
    data=xor_attk(xor1_dim,nchall,threshold,n_generations,model_name,pop_size,alphas,noise,xor)

     # Extract data from the XOR PUF attack results
    C_filtered = jnp.array(data.get("xor_filt_chall"),dtype=jnp.int32)
    lrn_r=jnp.array(data.get("lrn_r"),dtype=jnp.int32)
    C2_R=jnp.array(data.get("C2_R"),dtype=jnp.int32)
    C2=jnp.array(data.get("C2"),dtype=jnp.int32)


    # Transform challenges based on learned responses
    new_challenges=transform_flip(C_filtered,lrn_r)
    new_filtered_challenges=new_challenges

    global c_sample
    c_sample = generate_challenges(new_key(), (5000, new_challenges.shape[1]))

    # Compute true responses of the XOR PUF to filtered challenges
    # individual shape is now (N, k) -- true_r_individual[:, j] is arbiter j
    true_r_individual,true_r = xor(new_filtered_challenges)
    true_r=true_r.flatten()
 
    # Generate validation challenges
    val_chall = filter_challenges(                                      
        xor, generate_challenges(new_key(), (25_000, (dim))), threshold   
    )

    # Initialize accuracy variable
    overall_acc=0.5
    # Iterate until desired accuracy is achieved
    while overall_acc<0.98:
        # Define attack arguments
        attk_args = dict(
        pop_size=pop_size,
        n_generations=n_generations,
        model_name=model_name,
        threshold=threshold,
    )
        
        # Perform PUF attack(Right XOR PUF)
        lrn_w,best_fitness,total_time=attack_indv_puf(new_filtered_challenges,attk_args,noise,xor,nxor,alphas)

        # Calculate accuracy on validation challenges
        # individual shape is now (N, k) -- access arbiter j as individual[:, j]
        val_old_xor_r_individual, val_old_xor_r=xor(val_chall)
        val_old_xor_r=val_old_xor_r.flatten()

        lrn_r_individual,lrn_r = xor_get_response(lrn_w, val_chall) 
        lrn_r=lrn_r.flatten()

        acc = jnp.equal(val_old_xor_r, lrn_r).mean() 
        print("Accuracy",acc)
        overall_acc = jnp.array([acc, 1 - acc]).max()
           
    indv_acc=[]

    # Iterate over the XOR instances.
    # individual is (N, k): column j holds all of arbiter j's responses.
    for i in range(nxor):
        lrn_r_subset = lrn_r_individual[:, i]

        # Iterate over all arbiter columns of the true individual responses
        for j in range(nxor):
            r_subset = val_old_xor_r_individual[:, j]

            # Calculate and print the accuracy
            this_acc = jnp.equal(r_subset, lrn_r_subset).mean()
            print(f"Prediction acc wv_{i + 1} vs wv_{j + 1}", this_acc)
            indv_acc.append(this_acc)


    # Print and return results and data
    print(f"nchall={nchall}: {overall_acc.mean()}")

    # Calculate final fitness
    final_fitness = fitness_chall_filter( 
        new_key(), lrn_w, new_filtered_challenges, threshold
    ).item()

    # Create a dictionary to store collected data
    data = dict(
        overall_acc=overall_acc,
        model_name=model_name,
        dim=dim,
        pop_size=pop_size,
        n_generations=n_generations,
        threshold=threshold,
        nchall=nchall,
        alphas=alphas,
        final_fitness=final_fitness,
        noise=noise,
        puf_type=f"xor {xor.dim}",
        weight_hash=hash_w(xor.weight),
        nxors=nxor,
        best_fitness=best_fitness,
        total_time=total_time,
        indv_acc=indv_acc
    )

    # Add alpha values to the data dictionary
    for i in range(len(alphas)):
        data[f"alpha{i+1}"] = alphas[i]

    # Fill remaining alpha slots in the data dictionary with None
    for i in range(len(alphas), 5):
        data[f"alpha{i+1}"] = None

    # Return XOR PUF, learned XOR PUF, attack states, and collected data
    print("TOTAL TIME IS",total_time)
    return data



'''
this code is designed to conduct experiments to test the accuracy of noisy XOR PUFs under various conditions,
including different dimensions, target accuracy values, and noise levels. The results are recorded in a CSV file for further analysis.
'''
def noisy_challenge_filter_test(
    dims=[(3, 128), (4, 128), (5, 128), (6, 128)],
    targets=[0.8, 0.85, 0.9, 0.95, 0.99],
    nsamples=25,
    threshold=1,
    data_path="data/noise_filter.csv",
):
    
    """
    Conducts experiments to test the accuracy of noisy XOR PUFs under various conditions.

    Args:
        dims (list): Dimensions of the XOR PUFs to be tested. Each element should be a tuple representing the number of XOR gates and the number of bits.
        targets (list): Target accuracy levels to be tested.
        nsamples (int): Number of experiments to be conducted for each combination of parameters.
        threshold (float): Threshold value for filtering challenges.
        data_path (str): Path to the CSV file where experimental results will be recorded.

    Returns:
        list: List containing experimental data collected from the tests.
    """

    import csv

    # Initialize an empty list to store experimental results
    data = []

    # Initialize a trial counter
    trial = 1

    # Loop over different PUF dimensions
    for dim in dims:
        print(f"trial {trial}")
        trial += 1

        # Loop over target accuracy levels
        for target in targets:

            # Perform multiple experiments (specified by nsamples)
            for _ in range(nsamples):
                # Create a new XOR-based PUF with the specified dimensions
                puf = Xor(new_key(), dim)

                # Generate a set of challenges for testing
                c = generate_challenges(new_key(), (50_000, (puf.dim[1])))

                # Use jax.vmap to calculate the noise factor for each weight
                get_factor = jax.vmap(
                    lambda w: target_error(
                        new_key(), w, c, target=target, nsamples=1_000
                    )
                )

                # Compute the noise factor (sigma_error) for the current PUF's weight
                sigma_error = get_factor(puf.weight).flatten()

                # Generate validation challenges
                val_chall = generate_challenges(new_key(), (50_000, (puf.dim[1])))

                # Compute the noisy responses of the PUF
                _, noisy_r = noisy_xor_get_response(
                    new_key(), puf.weight, val_chall, sigma_error
                )
                noisy_r = noisy_r.flatten()

                # Calculate the accuracy of the noisy PUF
                xor_accuracy = (noisy_r == puf(val_chall).flatten()).mean()

                # Create a dictionary to store experimental data
                run_data = dict(
                    dim=dim[1],
                    nxor=dim[0],
                    xor_accuracy=xor_accuracy,
                    threshold=threshold,
                    target=target,
                )

                # Initialize placeholders for accuracy and sigma values
                for i in range(3, 7):
                    run_data[f"p{i+1}_accuracy"] = None
                    run_data[f"sigma{i+1}"] = None

                # Calculate and store accuracy and sigma values for individual PUFs
                for i in range(dim[0]):
                    arb_w = puf.get_weight(i)
                    arb_resp = get_response(arb_w, c).flatten()
                    noise_resp = noisy_get_response(
                        new_key(), arb_w, c, sigma_error[dim[0] - i]
                    ).flatten()
                    run_data[f"p{i+1}_accuracy"] = float(
                        (arb_resp == noise_resp).mean()
                    )
                    run_data[f"sigma{i+1}"] = sigma_error[i]

                # Append the experimental data to the list
                data.append(run_data)

    # Write the experimental data to a CSV file
    with open(data_path, "a+") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

    # Return the collected data
    return data