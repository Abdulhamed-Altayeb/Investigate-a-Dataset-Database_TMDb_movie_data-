#!/usr/bin/env python
# coding: utf-8

# # Project: Investigate a Dataset (Database_TMDb_movie_data)
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# <li><a href="#conclusions">Conclusions</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# This data set contains information
# about 10,000 movies collected from
# The Movie Database (TMDb),
# including user ratings and revenue.
# 
# ● Certain columns, like ‘cast’
# and ‘genres’, contain multiple
# values separated by pipe (|)
# characters.
# 
# ● There are some odd characters
# in the ‘cast’ column. Don’t worry
# about cleaning them. You can
# leave them as is.
# 
# ● The final two columns ending
# with “_adj” show the budget and
# revenue of the associated movie
# in terms of 2010 dollars,
# accounting for inflation over
# time.

# ## What do we want to infer from the data?
# Which genres are most popular from year to year?
# What kinds of properties are associated with movies that have high revenues?

# In[78]:


#importing all packages 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling
# in data wrangling we will load data file and we will know about thier properties. 
# ### General Properties

# In[79]:


# Load data amd showing data fram 
df= pd.read_csv("tmdb-movies.csv")
df.head()


# In[80]:


# showing tuple of the dimensions of the dataframe
df.shape


# The data contains (10866 rows) and (21 columns)
# 

# In[81]:


#showing info about dataset 
df.info()


# There are many missing data 
# and we know data types for data

# In[82]:


#showing missing values 
df.isnull().sum()


# imdb_id , cast, homepage,director ,tagline ,keywords , overview ,genres and production_companies have missing values

# In[83]:


#showing is there duplicates 
df.duplicated().sum()


# There is one duplicated data

# In[84]:


#showing the datatypes of the columns
df.dtypes


# In[85]:


#showing description of dataset 
df.describe().round(2)


# ●Min value of budget, revenue, runtime, budget_adj and revenue_adj = 0
# 
# ●Mean value of budget 1.462570e+07 and max 4.250000e+08	
# 
# ●Mean value of revenue 3.982332e+07 and  max 2.781506e+09
# 
# 
# 

# ### Data Cleaning 

# In[86]:


#remove duplicated values
df.drop_duplicates(inplace=True)


# In[87]:


#make sure we dont have duplicated values
df.duplicated().sum()


# As shown we dont have any duplicated values

# In[88]:


# remove columns that we dont need
df.drop(["imdb_id","cast","tagline","keywords","director","homepage","overview","production_companies"], axis=1, inplace=True)
df.head(0)


# Removing the columns that we dont need at our analysis

# In[89]:


# convert release_date to datetime format
df['release_date'] = pd.to_datetime(df['release_date'])


# In[90]:


# make sure we cinvert release_date
df.dtypes
df['release_date'].head()


# As shown release_date converted to datetime

# In[91]:


# get budget mean
budget_mean = round(df['budget'].mean())
print(budget_mean)


# In[92]:


# replaced budget zero values with mean value
df['budget'] = df.budget.mask(df.budget == 0,budget_mean)
df.loc[df['budget'] == 0]


# we replaced budget zero values with mean value As shown 

# In[93]:


# revenue mean
revenue_mean = round(df['revenue'].mean())
print(revenue_mean)


# In[94]:


# replaced revenue zero values with mean value
df['revenue'] = df.revenue.mask(df.revenue == 0,revenue_mean)
df.loc[df['revenue'] == 0]


# we replaced revenue zero values with mean value As shown 

# In[95]:


#get runtime mean
runtime_mean = round(df['runtime'].mean())
print(runtime_mean)


# In[96]:


# replaced runtime zero values with mean value
df['runtime'] = df.runtime.mask(df.runtime == 0,runtime_mean)
df.loc[df['runtime'] == 0]


# we replaced runtime zero values with mean value As shown 

# In[97]:


df.describe()


# As shown all zero values replaced with mean values

# <a id='eda'></a>
# ## Exploratory Data Analysis
#  Now going to explore data computing statistics and visualizing data 
# ### General look

# In[98]:


df.hist(figsize=(15,10));


# ### Answer The first Q (Which genres are most popular from year to year?)

# In[99]:


# combine cells of genre column
def extract_data(genres):
    data = df[genres].str.cat(sep = '|')
    data= pd.Series(data.split('|')) 
    count= data.value_counts()   
    return count   


# In[100]:


#extract geners 
genres = extract_data('genres')
genres.head(20)


# In[101]:


genres.plot.bar()


# The figuer tell us the most popular genres is Drama ( 4760 ) then the second is Comedy (3793 ) then Thriller (2907 )

# ### Answer The second Q (What kinds of properties are associated with movies that have high revenues?)
# 

# In[102]:


#lets see count of movies over realese_year 
movie_released= df.groupby('release_year').count()['id']
movie_released.head(10)


# In[103]:


#visualize the result
plt.scatter(movie_released.index , movie_released)
plt.show()


# The figure tell us that from beging of years 2000 that up to 200 movies , and from 2010 it increadbl increase until it over 700 movis

# In[104]:


df.plot(x="runtime", y="revenue", kind="scatter")
plt.show()
df['runtime'].describe()


# max runtime value is too longt 900 min and 25% of movies have a runtime of 90 minutes. 50% of the movies have a runtime of 99 minutes, 75% have runtime over 111 minutes

# In[105]:


sns.heatmap(df.corr() , annot =True)


# From heat_map There is a strong correlation betweeen vote_count and revenue and budget and revenue
# Then not very strong correlation between (popularity and reveue )

# In[106]:


budgets = df.groupby('release_year').mean()['budget']
revenues = df.groupby('release_year').mean()['revenue']


# In[107]:


#visulaizing by scatter plot between revenue and budgets
plt.scatter(revenues.index , revenues)
plt.scatter(budgets.index , budgets)
plt.title('Budget and Revenue Yearly Comparison')
plt.xlabel('Year')
plt.ylabel('Budget and Revenue')
plt.show()


# Budgets has effect on revenue over years  because form figuer its positve relationship

# <a id='conclusions'></a>
# # Conclusions
# ● The most popular genres is Drama ( 4760 ) then the second is Comedy (3793 )
# 
# ● from beging of years 2000 that up to 200 movies released and from 2010 it increadble increase until it over 700 movis
# 
# ● max runtime value is too longt 900 min and 25% of movies have a runtime of 90 minutes. 50% of the movies have a runtime of 99 minutes, 75% have runtime over 111 minutes
# 
# ● There is a strong correlation betweeen vote_count and revenue and budget and revenue  
# 
# 

# ### limitations
# ● The budget and revenue currency is unk nown
# 
# 
# ●  Cant know the way of caculate vote_average 

# In[108]:


from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])


# In[ ]:




