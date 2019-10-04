#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
np.random.seed(123)
import pandas as pd


# In[4]:


# A bank found that the average number of cars waiting during the noon hour at a drive-up window 
# follows a Poisson distribution with a mean of 2 cars. 
# Make a chart of this distribution and answer these questions concerning the probability of cars 
# waiting at the drive-up window.


cars = stats.poisson(2)
sns.distplot(cars.rvs(1000))


# What is the probability that no cars drive up in the noon hour?


#least_1 = cars.sf(0)

# # What is the probability that 3 or more cars come through the drive through?

# least_3 = cars.sf(2)

# # How likely is it that the drive through gets at least 1 car?

# p_at_least_1 = cars.sf(0)


wth = np.random.poisson(2, 10000)

number_of_cars = pd.DataFrame(np.random.poisson(2, 10000))
(number_of_cars == 0).mean()
(number_of_cars >= 3).mean()
(number_of_cars >= 1).mean()


# In[79]:


# Grades of State University graduates are normally distributed with a mean of 3.0 and a standard deviation of .3. Calculate the following:

grades = stats.norm(3, .3)


# What grade point average is required to be in the top 5% of the graduating class?

top_5 = grades.ppf(.95)

# What GPA constitutes the bottom 15% of the class?

bottom_15 = grades.ppf(.15)
bottom_15


# An eccentric alumnus left scholarship money for students in the third decile from the bottom of their class. 
# Determine the range of the third decile. 
# Would a student with a 2.8 grade point average qualify for this scholarship?


scholarship = grades.ppf(.2)
scholarship

# If I have a GPA of 3.5, what percentile am I in?

h_gpa = grades.cdf(3.5)
h_gpa



trials = cols = 100

samples = rows = 100

mu, sigma = 3.0, .3

s_grades = np.random.normal(mu,sigma,trials * samples)
# s_grades = np.random.normal(mu,sigma,trials * samples).reshape(rows, cols)

sns.distplot(s_grades)

np.percentile(s_grades, 95)
# s_grades = pd.DataFrame(s_grades)

# # s_grades = s_grades.apply(lambda x: x.sort_values().values)


# s_grades.quantile(.95).sort_values(ascending= True) #3.34
# # s_grades.quantile(.15).sort_values(ascending = False) # 2.84
# # s_grades.quantile(.30).sort_values(ascending = False) # 2.94
# np.where(s_grades >= 3.34, True, False)
    
# len(np.where(s_grades >= 3.34, True, False))

# graph



                                             


# In[ ]:





# In[82]:


# A marketing website has an average click-through rate of 2%. 

traffic = stats.binom(4326, .02)

# One day they observe 4326 visitors and 97 click-throughs. 
# How likely is it that this many people or more click through?

clicks = traffic.sf(96)       

clicks

##### Experimental #####


n, p = 4326, .02  # number of trials, probability of each trial
c = np.random.binomial(n, p, 1_000_000)
c = sum(np.random.binomial(n, p, 1_000_000) > 96)/1_000_000
c

clicks, c


# In[83]:


# You are working on some statistics homework consisting of 100 questions where 
# all of the answers are a probability rounded to the hundreths place.
# Looking to save time, you put down random probabilities as the answer to each question.

# What is the probability that at least one of your first 60 answers is correct?


got_lucky = stats.binom(60, .01).sf(0)

got_lucky

##### Experimental #####


n, p = 60, .01  # number of trials, probability of each trial
s = np.random.binomial(n, p, 1_000_000)
s = sum(np.random.binomial(n, p, 1_000_000) > 0)/1_000_000

got_lucky, s


# In[84]:


# The codeup staff tends to get upset when the student break area is not cleaned up. 
# Suppose that there's a 3% chance that any one student cleans the break area when they visit it, 
# and, on any given day, about 90% of the 3 active cohorts of 22 students visit the break area. 

n_visitors = .9 * 3 * 22
clean = stats.binom(n_visitors, .03)

# How likely is it that the break area gets cleaned up each day? 

day = clean.sf(0)

# How likely is it that it goes two days without getting cleaned up? All week?

day_2 = clean.cdf(0) ** 2
week = clean.cdf(0) ** 5



##### Experimental #####

n, p = (.9 * 3* 22), .03  # number of trials, probability of each trial
c = np.random.binomial(n, p, 100_000)
c = sum(np.random.binomial(n, p, 100_000) > 0)/100_000
cc = c **2
ccccc = c ** 5

day, c


# In[22]:


# You want to get lunch at La Panaderia, but notice that the line is usually very long at lunchtime. 
# After several weeks of careful observation, you notice that the average number of people in line 
# when your lunch break starts is normally distributed with a mean of 15 and standard deviation of 3. 
# If it takes 2 minutes for each person to order, and 10 minutes from ordering to getting your food, 
# what is the likelihood that you have at least 15 minutes left to eat your food before you have to go back to class? 
# Assume you have one hour for lunch, and ignore travel time to and from La Panaderia.


line = stats.norm(15, 3)
people = (60 - 15 - 10) // 2
p15 = line.cdf(people)

p15


mu, sigma = 15, 3 # mean and standard deviation
p = np.random.normal(mu, sigma, 1_000)
np.random.normal(mu, sigma, 1_000) > 17
pp = sum(np.random.normal(mu, sigma, 1_000) < 17)/1_000
pp


# In[ ]:





# In[107]:


# import pandas as pd
# from env import host, user, password



# def get_db_url(username, hostname, password, db_name):
#     return f'mysql+pymysql://{username}:{password}@{hostname}/{db_name}'

# query = '''
#     select * from salaries
# WHERE to_date like '9999-%%-%%' 
# '''
# url = get_db_url(user,host,password,'employees')

# salaries = pd.read_sql(query,url)
# mean = salaries['salary'].mean()
# std = salaries['salary'].std()
# salary_dist = stats.norm(mean,std)
# under_60k = salary_dist.cdf(60_000)
# over_95k = salary_dist.sf(95_000)
# between_65k_and_80k = salary_dist.cdf(80_000) - salary_dist.sf(65_000)
# top_5 = salary_dist.isf(.05)


# In[ ]:




