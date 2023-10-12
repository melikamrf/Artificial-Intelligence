#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas
import numpy as np
import random
import time
import matplotlib.pyplot as plt 

Ttable = pandas.read_csv ('D:/هوش/CA/2/truth_table.csv')
num_inputs = Ttable.shape[1] - 1
num_gens = num_inputs - 1
maximum = len(Ttable.index)
best_score = []


# In[2]:


######### hyper parameter #############
pop_size = 100
pc = 0.8
def pm(g):
     return 1/num_gens - g*((1/num_gens) - (1/pop_size))/150
generations = 250
##############################


# In[3]:


class Individual:
    def __init__(self,gens):
        self.gens = gens
        self.fitness = self.fitness()
    def fitness(self):
        return (Ttable[Ttable['Output'] == output(Ttable,self.gens) ].shape[0])
    def fit(self):
        return self.fitness == maximum


# In[4]:


def out(in1,in2,gate):
      return {
        0: in1 & in2,
        1: in1 | in2,
        2: in1 != in2,
        3: ~(in1 & in2),
        4: ~(in1 | in2),
        5: in1 == in2
    }[gate]
    
def output(x,gens):
    output = out(x.iloc[:,0],x.iloc[:,1],gens[0])
    for i in range(1,num_gens):
        output = out(output,x.iloc[:,i+1],gens[i])
    return output


# In[5]:


def random_selection(population):
  
    total = sum(x.fitness for x in population)    
    prob = [x.fitness/total for x in population]
    selected = np.random.choice(population, size = pop_size, replace=True, p = prob) #FPS 
    return selected.tolist()
   


# In[6]:


def tournament(items, n, k = 3):
    best = []
    for i in range(n):
        candidates = random.sample(items, k)
        best.append(max(candidates, key=lambda x: x.fitness))
    return best


# In[7]:


def REPRODUCE(x , y):
    point = np.random.randint(num_gens - 1)
    c1 = x[:point+1] + y[point+1:]
    c2 = y[:point+1] + x[point+1:]
    return Individual(c1),Individual(c2)

def uniform_crossover(x,y):
    c1 = []; c2 = []
    for i in range(num_gens):
        
        prob = random.random()
        if prob < 0.5:
            c1.append(x[i])
            c2.append(y[i])
        else:
            c1.append(y[i])
            c2.append(x[i]) 
    return Individual(c1),Individual(c2)


# In[8]:


def mutate(child):
    g = np.random.randint(num_gens)
    if child.gens[g] < 3:
            child.gens[g] += 3
    else:
            child.gens[g] -= 3
    


# In[9]:


def remove_min(pop,k):
    for i in range(k):
        pop.remove(min(pop,key = lambda x:x.fitness))
        


# In[10]:


def add_random(k):
    p = np.random.randint(6,size = (k,num_gens))
    pop = [Individual(p[i]) for i in range(k)]
    return pop


# In[11]:


def GA(population):
  for g in range(generations):
    
        #print(g)       
        #for i in population:
           #print('gen',i.gens,'fit',i.fitness)

        mating = random_selection(population)
        mating = mating + add_random(4)
        random.shuffle(mating)
        new_pop = []
        
        for i in range(len(mating)//2):
            x = mating.pop()
            y = mating.pop()

            p = np.random.uniform(0, 1.0)
            ch1,ch2 = x,y
            if p < pc :
              ch1,ch2 = REPRODUCE(list(x.gens), list(y.gens))
            p = np.random.uniform(0,1.0)
           
            if (p < pm(g)):
                mutate(ch1)
                mutate(ch2)
            
            new_pop.append(ch1)
            new_pop.append(ch2)
            if ch1.fit():
                return ch1.gens
            elif ch2.fit():
                return ch2.gens
       # best_score.append(max(new_pop,key = lambda x:x.fitness).fitness)

        remove_min(new_pop,4) 
        population = new_pop
    


# In[12]:


##### init #####
p = np.random.randint(6,size = (pop_size,num_gens))
pop = [Individual(p[i]) for i in range(pop_size)]
#################


# In[13]:


s = time.time()
print(GA(pop))
e = time.time()
print(e - s)


# In[14]:


#plt.plot(best_score)
#plt.xlabel('Generation')
#plt.ylabel('Best score (% target)')
#plt.show()





