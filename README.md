
# Physically Unclonable Functions (PUFs)

Functional puf is a PUF attack library with two main modules: Pufs/FunctionalPuf.py, and Attack/FunAttack.py.

Pufs/FunctionalPuf.py contains all the I/0 functions and some commonly used helper functions. 
The Functional PUF Library is a Python implementation of Physically Unclonable Functions (PUFs) using functional representations. This library provides tools for generating challenges, calculating PUF responses, and simulating noisy environments for evaluation.

Functional Puf uses Jax which uses threefry counter PRNG for reproducible random number generation. 
EVERY time you need to generate a random number use `new_key()` which is a global infinite generator for subkeys.
It's best to pass an RNG key as a parameter rather then generating it inside a function. ESPECIALLY if that function is going to be jitted or vmapped.


The `Arbiter` and `Xor` classes encapsulate the IO functions but should not be used in any jitted vmaped style functions. Instead prefer 
`get_response` and `xor_get_response`.

```Python
from Attack import *
from Pufs import *
nchall = 10_000
arb = Arbiter(new_key(), (1,128))
xor = Xor(new_key(), (3, 128))
chall = generate_challenges(new_key(), (25_000, xor.dim[1]))
arb_response = arb(chall)
xor_response = xor(chall)
```


Attack/FunAttack is mostly a collection of fitness functions and attack plumbing. 
This section consists of code related to attacking XOR and Interpose PUFs.

#### XOR PUF Attack Parameters:
- `dim`: Tuple specifying the number of XOR gates and the dimension of each gate.
- `nchall`: Number of challenge-response pairs to generate and use in the attack.
- `threshold`: Threshold value used for filtering challenges.
- `n_generations`: Number of iterations for the optimization algorithm.
- `model_name`: Name of the optimization algorithm used (e.g., "CMA_ES").
- `pop_size`: Population size for the optimization algorithm.
- `alphas`: Tuning parameters for the optimization process.
- `noise`: Flag to enable noise in the attack.
- `rxor`: Flag to specify if it is an Interpose PUF being attacked.

#### Interpose PUF Attack Parameters:
- `xor1_dim`: Dimensions of the left XOR PUF.
- `dim`: Dimensions of the complete IPUF (including left and right XOR).
- `nchall`: Number of challenge-response pairs to generate and use in the attack.
- `threshold`: Threshold value used for filtering challenges.
- `n_generations`: Number of iterations for the optimization algorithm.
- `model_name`: Name of the optimization algorithm used (e.g., "CMA_ES").
- `pop_size`: Population size for the optimization algorithm.
- `alphas`: Tuning parameters for the optimization process.
- `noise`: Flag to enable noise in the attack.



The library assumes that **everything** is a row vector, for example in the xor above each row of xor.weights corresponds to the weight of a single consituent puf. 
This makes utilizing `jax.vmap` easier to reason about and allows the inner loop of the evolution strategy to run very quickly. 

# Attacks

The core attack function is `run_es_loop(rng, num_steps, fit_fn, model)` the fitness function should have signature
`fit_fn(rng, w)` even if it doesn't need an rng. Often this means encapsulating in a lambda or some other function.

for example here's how you might change a fitness function

```Python

def _x3_fit(rng, w, w1, w2, valid_chall, threshold, alpha):
    """
    Calculates a fitness score for a given weight vector w based on challenges, comparisons with previous weights,
    and a penalty term. The penalty term penalizes weights that do not perform well compared to previous weights. 
    The final fitness score is a combination of the filtered fitness score and the scaled penalty.

    Args:
        rng (numpy.ndarray): Random number generator.
        w (numpy.ndarray): Weight vector.
        w1 (numpy.ndarray): Additional parameter w1.
        w2 (numpy.ndarray): Additional parameter w2.
        valid_chall (numpy.ndarray): Matrix of valid challenges.
        threshold (float): Threshold for filtering challenges.
        alpha (float): Penalty coefficient.

    Returns:
        float: Final fitness score.
    """
    #Write code to return fitness based on certain conditions. Example : xorn_fitness function

# define w1, w2, valid_chall, threshold, and alpha here

x3_fit = lambda rng, w: _x3_fit(rng, row_vec(w), w1, w2, valid_chall, threshold, alpha)
x3_fit = jax.vmap(x3_fit, in_axes=(None, 0))
```

reccomended: **https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html**

# Simple Flow Diagram

[XOR PUF] --> [Seed Generation] --> [PRNG] --> [Generate Weights] -->  [Generate 1 Weight] --> [return XOR weights]
    
[Generate New Key] --> [Generate Challenges] --> [Initialize Challenge Filter] --> [XOR Filtered Challenges]
  
[XOR Filtered Challenges] --> [XOR Weights] --> [XOR Response]
   
[X1 Attack] --> [Run ES Loop] --> [Fitness Calculation (X1)] --> [Update Learned Weights] 
   
[XN Attack] --> [Run ES Loop] --> [Fitness Calculation (Xn)] --> [Fitness direct comparison] --> [Update Learned Weights] 
   
[Generate Validation Challenges]
 
[Calculate Learned Response] --> [Accuracy Calculation] --> [Results]

##### Step-by-Step Explanation for XOR PUF Attack
1. Initialization:
    Seed Generation: A seed is randomly chosen between 1 to 1000.
    PRNG (Pseudo Random Number Generator): Utilizes the seed to generate keys and subkeys.
    Weight Generation: Based on the number of XORs (nxors) and dimensions (dims), weights are generated.
2. Challenge Generation:
    Generate New Key: A new key is generated to initiate the challenge generation process.
    Generate Challenges: Random challenges consisting of 1s and -1s are generated with dimensions (nchall, puf.dim[1]).
    Initialize Challenge Filter: Filters the generated challenges until the number of challenges (nchall) is met.
3. Response Calculation:
    XOR Filtered Challenges: Challenges are filtered based on a threshold.
    Get Response: The weight matrix is multiplied by the transposed challenge matrix to convert the results into 0s and 1s.
    Perform Bitwise XOR: Bitwise XOR operation is performed on the response bits from nxors.
4. Attack Simulation (X1 Attack and XN Attack):
    Learning Phase:
    Run Evolutionary Strategy (ES) Loop: Multiple iterations are performed to optimize the weight vectors.
    Fitness Calculation: Fitness of the weight vectors is calculated based on the challenges, previous weights, and a penalty term.
    Update Learned Weights: Weights are updated for each alpha value using the attack model.
5. Validation:
    Generate Validation Challenges: Similar to the initial challenge generation but used for validating the learned model.
    Calculate Learned Response: Responses are calculated for the validation challenges.
    Accuracy Calculation: The output from the XOR operation on validation challenges is compared with the true response to calculate accuracy.
6. Results:
    Overall Accuracy: The accuracy of the learned model is evaluated.
    Fitness: The final fitness score is reported.



##### Step-by-Step Explanation of Interpose PUF Attack
1. Initialization:
    Seed Generation: A random seed between 1 to 1000 is selected.
    PRNG (Pseudo Random Number Generator): The seed is used to generate keys and subkeys.
    Weight Generation: Weights for the L_XOR PUF are generated.
2. Challenge Generation:
    Generate Challenges (C): Random challenges consisting of 1s and -1s are generated with dimensions (nchall, puf.dim[1]).
    Threshold-Based Filtering (C_1): Challenges undergo filtering based on a specified delay threshold to create a refined set, C_1.
3. Response Calculation:
    Calculate Responses (R_1): Using the L_XOR PUF, responses (R_1) are generated for the filtered challenges (C_1).
    Transform Challenges (C_2): Challenges in C_1 are transformed into a new set, C_2, using the responses R_1.
4. Refinement of Challenges (C_2_filtered):
    Threshold-Based Filtering (C_2_filtered): Challenges in C_2 undergo further filtering based on the delay threshold of the R_XOR PUF, resulting in a refined set, C_2_filtered.
5. Reverse Transformation and Restoration:
    Reverse Transformation: A reverse transformation is applied to C_2_filtered to restore it to its original form, referred to as C_filtered.
6. Modeling and Derivation of Responses:
    Construct L_XOR Model (L_XOR_modelled): A model of the L_XOR PUF (L_XOR_modelled) is constructed using challenges from C_filtered, and corresponding responses (R_learned_left) are derived.
    Transformation of Challenges (C_filtered_new): Challenges within C_filtered are transformed using the responses R_learned_left, resulting in a new set, C_filtered_new.
    Modeling R_XOR (R_XOR_modelled): Using challenges from C_filtered_new, a model of the R_XOR PUF (R_XOR_modelled) is constructed.
7. Attack Execution for Left and Right XOR:
    Modeling L_XOR: The L_XOR_modelled is crafted using CMAES, computing fitness values, and applying a penalty parameter alpha for subsequent instances of the Left XOR.
    Transformation of Challenges: The response of L_XOR_modelled is strategically added at the middle index of the challenge, based on defined criteria.
    Modeling R_XOR: The R_XOR_modelled is constructed using CMAES, computing fitness values, and applying the penalty parameter alpha for further instances of the Right XOR.
8. Conclusion:
    The modeling process, along with the transformations involved, plays a pivotal role in successfully attacking the interpose PUF by understanding and influencing the behavior of XOR PUFs tailored for an interpose PUF attack.
<!-- Generate new key:
    Generate 1 weight
    Generate challenges
    Filter challenges
    XOR filtered challenges
    Calculate delta response
    Calculate delta delay value
    Create accept criteria
    Get response
    Convert to values - 0 and 1
    Perform bitwise XOR on response bits from n xors
X1 ATTACK:
    Run ES loop
    Fitness calculation
    Update learned weights
XN ATTACK:
    Specify model and xnfit fn
    Run ES loop
    Fitness calculation
    Update learned weights
Validation:
    Generate validation challenges
    Initialize challenge filter
    Calculate learned response
    Perform bitwise XOR on response bits from n xors
    Calculate fitness
    Fitness direct comparison
    Update learned weights
Results:
    Overall accuracy
    Fitness -->
# How to Run the Code

Download the folder. 
Use pip install -r requirements.txt to download all the requirements.

1. Open the command prompt.
2. Navigate to the directory where your Python script is located using the cd command. 
3. Once you're in the correct directory, you can run the script with the desired attack type argument. For example, to perform an XOR PUF attack:
    python main.py xor

This command tells Python to executemain.py and pass the argument 'xor' to it, indicating that you want to perform an XOR PUF attack. Similarly, you can replace 'xor' with 'ipuf' if you want to perform an iPUF attack.

To change the parameters of the function xor_attk()  or ipuf_attk() function in the FunAttack.py file in the Attack Folder for performing different attacks with different thresholds and number of challenges.
For example, if you want to change the parameters of the xor_attk() function, find its definition in FunAttack.py and modify the parameters:

```Python
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
)
```
Adjust the parameters as needed:
dim: Change the number of XORs by modifying the first element of the tuple. For example, to change the number of XORs to 4, set dim=(4, 128).
nchall: Adjust the number of challenges as needed. For instance, if you want to use 20000 challenges, set nchall=20000.
threshold: Change the threshold value for the attack. You can set it to any desired value between 0 and 1.
Similarly, you can modify the parameters of the ipuf_attk() function following the same steps.


## Acknowledgments

We would like to express our gratitude to [Name] for their valuable contributions and support in the development of the Functional PUF Library. Special thanks to [Name] for their insightful feedback and suggestions during the implementation process.

## Author

Haytham Idriss, Furqan Khan, [Name], [Name].

For inquiries or feedback, please contact hidriss@pfw.edu or furqank8@gmail.com.
