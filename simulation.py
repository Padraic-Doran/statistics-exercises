#!/usr/bin/env python
# coding: utf-8

# In[39]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
np.random.seed(29)

n_trials = nrows = 100
n_dice = ncols = 2

rolls = np.random.choice([1, 2, 3, 4, 5, 6], n_trials * n_dice).reshape(nrows, ncols)


# In[9]:


dice = pd.DataFrame(np.random.randint(1,7,(1000000,2)))
doubles = (dice_sample[0] == dice_sample[1]).mean()
doubles


# In[11]:


# If you flip 8 coins, what is the probability of getting exactly 3 heads? What is the probability of getting more than 3 heads?
coins = pd.DataFrame(np.random.randint(0,2,(1000000,8)))
p_three_heads = (coins.sum(axis=1) == 3).mean()
p_over_three = (coins.sum(axis=1) > 3).mean()
p_three_heads
p_over_three


# In[49]:


# There are approximitely 3 web development cohorts for every 1 data science cohort at Codeup. 
# Assuming that Codeup randomly selects an alumni to put on a billboard, 
# what are the odds that the two billboards I drive past both have data science students on them?

billboards = np.random.random((1000000,2))
ptds = (billboards <= .25).prod(axis=1).mean()


# In[21]:


# Codeup students buy, on average, 3 poptart packages (+- 1.5) a day from the snack vending machine. 
# If on monday the machine is restocked with 17 poptart packages, 
# how likely is it that I will be able to buy some poptarts on Friday afternoon?

poptarts = np.random.normal(3, 1.5,(1000000,5))
poptarts = poptarts.round()
p_friday = (poptarts.sum(axis=1) <= 16).mean()
p_friday


# In[26]:


# Compare Heights

# Men have an average height of 178 cm and standard deviation of 8cm.
# Women have a mean of 170, sd = 6cm.
# If a man and woman are chosen at random, P(woman taller than man)?

men = np.random.normal(178, 8, 1000000)
men
# women = np.random.normal(170, 6, 1000000)
# ptallerwoman = (women > men).mean()
# ptallerwoman


# In[75]:


# When installing anaconda on a student's computer, there's a 1 in 250 chance 
# that the download is corrupted and the installation fails. 

# What are the odds that after having 50 students download anaconda, no one has an installation issue? 100 students?

# What is the probability that we observe an installation issue within the first 150 students that download anaconda?

# How likely is it that 450 students all download anaconda without an issue?

fail = 1/250
fail_fifty = (np.random.random((10000,50)) > fail).prod(axis = 1).mean()
fail_100 = (np.random.random((10000, 100)) > fail).prod(axis=1).mean()
fail_150 = 1 -(np.random.random((10000, 150)) > fail).prod(axis=1).mean()
fail_450 = (np.random.random((10000, 450)) > fail).prod(axis=1).mean()

fail_fifty


# In[74]:


# There's a 70% chance on any given day that there will be at least one food truck at Travis Park.
# However, you haven't seen a food truck there in 3 days. How unlikely is this?

# How likely is it that a food truck will show up sometime this week?

three_days = (np.random.random((10000,3)) > .7).prod(axis=1).mean()
three_days
week = (np.random.random((10000,7)) <= .7).prod(axis=1).mean()
week
three_days


# In[72]:


# If 23 people are in the same room, what are the odds that two of them share a birthday? 
# What if it's 20 people? 40?
import pandas as pd

birthday_23 = pd.DataFrame(np.random.randint(0,365,(10_000,23)))
share_birthday_23 = birthday_23.apply(lambda x: len(set(x)) != len(x),axis=1).mean()
share_birthday_23
birthday_20 = pd.DataFrame(np.random.randint(0,365,(10_000,20)))
share_birthday_20 = birthday_20.apply(lambda x: len(set(x)) != len(x),axis=1).mean()
share_birthday_20
birthday_40 = pd.DataFrame(np.random.randint(0,365,(10_000,40)))
share_birthday_40 = birthday_40.apply(lambda x: len(set(x)) != len(x),axis=1).mean()
share_birthday_40


# In[ ]:




