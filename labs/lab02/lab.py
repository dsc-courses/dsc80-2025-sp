# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def trick_me():
    # create cleaned 
    col_names = ['Name','Name','Age']
    tricky_1 = (pd.DataFrame([['Brain','Mary',45],['Jen','Paul',15],['Kenny','Joseph',27],['Sue','Ally',65],['Aria','Harry',33]],
                         columns=col_names))

    # tricky_1.iloc[:,2]
    tricky_1.to_csv('tricky_1.csv',index=False)

    try:
        tricky_2 = pd.read_csv('tricky_1.csv')
    except Exception as e:
        return 3

    if tricky_2.columns.tolist() == tricky_1.columns.tolist():
        return 2
    else:
        return 3


def trick_bool():
    col_names = [True,True,False,False]
    bools = pd.DataFrame([[2,4,6,8],[3,5,7,9],[4,8,12,16],[5,10,15,20]],columns=col_names)

    return [4,10,13]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def population_stats(cleaned):

    new_cleaned = pd.DataFrame(columns=['num_nonnull','prop_nonnull','num_distinct','prop_distinct'],index = cleaned.T.index.tolist())

    numnotnull = cleaned.notnull().sum().tolist()
    propnotnull = np.array(numnotnull) / cleaned.shape[0]
    numdistinct = cleaned.nunique().tolist()
    propdistinct = np.array(numdistinct) / np.array(numnotnull)

    new_cleaned.iloc[:,0] = numnotnull
    new_cleaned.iloc[:,1] = propnotnull
    new_cleaned.iloc[:,2] = numdistinct
    new_cleaned.iloc[:,3] = propdistinct

    return new_cleaned


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_common(cleaned, N=10):
    new_cleaned = pd.DataFrame(index=np.arange(N))

    for i in range(cleaned.shape[1]):
        name_col = cleaned.columns[i] + '_values'
        count_col = cleaned.columns[i] + '_counts'

        value_counts = cleaned.iloc[:,i].value_counts()
        # returns a Series with index: unique values and col: counts of the value

        amount_nan = N - len(value_counts.index[:N])

        if amount_nan > 0: # not enough distinct values bc N > num distinct
            value_counts.index[:N] = value_counts.index[:N].tolist + [np.nan] * (amount_nan)
            value_counts.values[:N] = value_counts.values[:N].tolist + [np.nan] * (amount_nan)

        new_cleaned = new_cleaned.assign(**{name_col : value_counts.index[:N]})
        new_cleaned = new_cleaned.assign(**{count_col : value_counts.values[:N]})
    
    return new_cleaned


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def super_hero_powers(powers):
    powers = powers.set_index('hero_names')

    name_most_power = powers.sum(axis=1).sort_values(ascending=False).index[0] # sum of the rows --> add up all columns for each row  

    after_flight = powers[powers['Flight']==True].sum(axis=0).sort_values(ascending=False).index[1] # sum of columns --> add up all rows for each col

    # first: sum of the rows and filter cleaned which superheroes have 1 power 
    # second: sum the columns and find the superpower that is most common among superheroes with only 1 power 
    name_one_power = powers[powers.sum(axis=1) == 1].sum(axis=0).sort_values(ascending=False).index[0]

    return [name_most_power,after_flight,name_one_power]

# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def clean_heroes(heroes):
    missing_str = ['-','Unknown','','None',-99.0]
    new_cleaned = heroes.replace(to_replace=missing_str,value=np.nan)
    return new_cleaned


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def super_hero_stats():
    return ['Onslaught', 'George Lucas', 'bad', 'Marvel Comics', 'NBC - Heroes', 'Groot']


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def clean_universities(cleaned):
    cleaned['institution'] = cleaned['institution'].str.replace('\n',', ')
    cleaned['broad_impact'] = cleaned['broad_impact'].astype(int)

    # split 'national_rank' column
    cleaned['nation'] = cleaned['national_rank'].str.split(', ').str[0]
    cleaned['national_rank_cleaned'] = cleaned['national_rank'].str.split(', ').str[1]
    cleaned['national_rank_cleaned'] = cleaned['national_rank_cleaned'].astype(int)

    # cleaned['nation'].unique() 
    # UK and United Kingdom; USA and United States; Czechia and Czech Republic 
    cleaned['nation'] = cleaned['nation'].replace(['UK','USA','Czechia'],['United Kingdom','United States','Czech Republic'])
    cleaned = cleaned.drop('national_rank',axis=1)

    cleaned['is_r1_public'] = (
        (cleaned['control']=='Public') &
        cleaned['city'].notna() &
        cleaned['state'].notna()
        )
    
    return cleaned

def university_info(cleaned):
    # ----------------------------------------- Part 1 ------------------------------------------------
    states_cleaned = cleaned.groupby('state').count()[['institution']]
    states_series = (states_cleaned[states_cleaned['institution'] > 3])['institution']
    states_3 = states_series.index.tolist()

    lowest_score = (cleaned[cleaned['state'].isin(states_3)]).groupby('state')['score'].mean().sort_values(ascending=True).index[0]
    # ---------------------------------------- Part 2 -----------------------------------------------
    mini_cleaned = cleaned.set_index('institution')[['quality_of_faculty','world_rank']]
    one_filter_cleaned = mini_cleaned[(mini_cleaned['world_rank'] <= 100)]
    both_filters_cleaned = mini_cleaned[(mini_cleaned['world_rank'] <= 100) & (mini_cleaned['quality_of_faculty'] <= 100)]

    prop_ranking = both_filters_cleaned.shape[0]/one_filter_cleaned.shape[0]
    # ---------------------------------------- Part 3 ----------------------------------------------
    mini_cleaned2 = cleaned.groupby(['state','is_r1_public'])[['institution']].count().reset_index(level='is_r1_public')
    private_states_series = mini_cleaned2[mini_cleaned2['is_r1_public']==False]['institution']

    total_state_series = cleaned[cleaned['state'].isin(private_states_series.index.tolist())].groupby('state')['institution'].count()

    num_states = np.count_nonzero((private_states_series/total_state_series).values >= 0.5)
    # --------------------------------------- Part 4 ----------------------------------------------
    mini_cleaned3 = cleaned.set_index('institution')[['world_rank','national_rank_cleaned']]
    mini_cleaned3 = mini_cleaned3.sort_values(['world_rank','national_rank_cleaned'],ascending=[False,True])
    top_national_cleaned = mini_cleaned3[mini_cleaned3['national_rank_cleaned']==1]

    lowest_in_its_nation = top_national_cleaned.index[0]

    return [lowest_score,prop_ranking,num_states,lowest_in_its_nation]

