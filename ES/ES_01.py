"""
The Evolution Strategy can be summarized as the following term:
{mu/rho +, lambda}-ES
Here we use following term to find a maximum point.
{n_pop/n_pop + n_kid}-ES
Visit my tutorial website for more: https://morvanzhou.github.io/tutorials/
"""
import numpy as np
import matplotlib.pyplot as plt

DNA_SIZE = 1 #the length of DNA
DNA_BOUND = [0,0.5] #the limit of value of DNA
N_GENERATIONS = 200 #the iteration of EA
POP_SIZE = 100 #the population size
N_KID = 50 #the kid born per generation

def fun(x):return np.sin(10*x)*x + np.cos(2*x)*x
    # 

def get_fitness(pred):
    return pred.flatten()
    # todo

def make_kid(pop, n_kid):
    # generate a empty kid holder 
    kids = {'DNA':np.empty((n_kid, DNA_SIZE))}
    kids['mut_strength'] = np.empty_like( kids['DNA'])

    for k1, k2 in zip(kids['DNA'], kids['mut_strength']):
        # crossover to make kids
        p1, p2 = np.random.choice(
                            np.arange(POP_SIZE),
                            size=2, 
                            replace=False)
        #从0到POPSIZE区间选择size个数，互不相同
    
        cp = np.random.randint(0, 2, DNA_SIZE, dtype = np.bool)
        # choose false or true (0 or 1)，选择DNAsize个数，0到1区间，以布尔类型给出
    
        #进行交配复制
        k1[cp] = pop['DNA'][p1, cp] # DNA from parent of p1
        k1[~cp] = pop['DNA'][p2, ~cp] # DNA from parent of p2
        k2[cp] = pop['mut_strength'][p1, cp]
        k2[~cp] = pop['mut_strength'][p2, ~cp]

        #进行变异
        k2[:] = np.maximum(k2 + (np.random.randn(*k1.shape)-0.5), 0.0)
        k1 += k2*np.random.randn(*k1.shape)
        k1[:] = np.clip(k1, *DNA_BOUND)

    return kids

def kill_bad(pop, kids):
    # put pop and kids together
    for key in ['DNA', 'mut_strength']:
        pop[key] = np.vstack((pop[key], kids[key]))

    fitness = get_fitness(fun(pop['DNA']))            # calculate global fitness
    idx = np.arange(pop['DNA'].shape[0])
    good_idx = idx[fitness.argsort()][-POP_SIZE:]   # selected by fitness ranking (not value)
    for key in ['DNA', 'mut_strength']:
        pop[key] = pop[key][good_idx]
    return pop

pop = dict(DNA=5 * np.random.rand(1, DNA_SIZE).repeat(POP_SIZE, axis=0),   # initialize the pop DNA values
           mut_strength=np.random.rand(POP_SIZE, DNA_SIZE))                # initialize the pop mutation strength values
print(pop)

plt.ion()       # something about plotting
x = np.linspace(*DNA_BOUND, 200)
plt.plot(x, fun(x))

for _ in range(N_GENERATIONS):
    # something about plotting
    if 'sca' in globals(): sca.remove()
    sca = plt.scatter(pop['DNA'], fun(pop['DNA']), s=200, lw=0, c='red', alpha=0.5); plt.pause(0.05)

    # ES part
    kids = make_kid(pop, N_KID)
    pop = kill_bad(pop, kids)   # keep some good parent for elitism

plt.ioff(); plt.show()
 


















