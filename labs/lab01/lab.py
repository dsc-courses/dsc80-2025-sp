# lab.py


from pathlib import Path
import io
import pandas as pd
import numpy as np
np.set_printoptions(legacy='1.21')


# ---------------------------------------------------------------------
# QUESTION 0
# ---------------------------------------------------------------------


def consecutive_ints(ints):
    if len(ints) == 0:
        return False

    for k in range(len(ints) - 1):
        diff = abs(ints[k] - ints[k+1])
        if diff == 1:
            return True

    return False


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def median_vs_mean(nums):
    sorted_nums = sorted(nums)

    # even length case 
    middle_index = len(sorted_nums)//2  # round down 
    if len(sorted_nums) % 2 == 0:
        middle = (sorted_nums[middle_index] + sorted_nums[middle_index-1])/2
    else:
        middle = sorted_nums[middle_index]

    avg = sum(nums)/len(nums)
    if middle <= avg: 
        return True 
    else:
        return False 


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def n_prefixes(s, n):
    new_str = ''
    for i in range(n,0,-1):
        new_str += s[0:i]
    return new_str


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def exploded_numbers(ints, n):
    new_ls = []

    num_padding = int(len(str(max(ints)+n)))

    for i in range(len(ints)):
        new_str = ''
        lower_bound = ints[i] - n 
        upper_bound = ints[i] + n 

        for j in range(lower_bound, ints[i], 1):    # start at lower bound and inc by 1 until middle number
            new_str += (str(j)).zfill(num_padding) + ' '

        for k in range(ints[i],upper_bound+1,1):
            new_str += (str(k)).zfill(num_padding) + ' '

        new_str = new_str.rstrip()

        new_ls.append(new_str)

    return new_ls


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def last_chars(fh):
    new_str = ''
    all_lines = []
    for line in fh:
        all_lines.append(line.strip())
        new_str += line.strip()[-1]
    return new_str


# ---------------------------------------------------------------------
# QUESTION 5
# ---------------------------------------------------------------------


def add_root(A):
    new_arr = np.arange(len(A))
    sqrt_arr = A + np.sqrt(new_arr)
    return sqrt_arr

def where_square(A):
    sqrt_arr = np.sqrt(A)
    bool_arr = (sqrt_arr == np.floor(sqrt_arr))
    return bool_arr


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_loop(matrix, cutoff):
    greater_col = []
    
    for col in range(matrix.shape[1]):
        # iterate number of columns times  
        col_sum = 0

        for row in range(matrix.shape[0]):
            # iterate number of rows times 
            # sum through all rows OF GIVEN col 
            col_sum += matrix[row][col]


        if col_sum/matrix.shape[0] > cutoff:    # divide by matrix.shape[0] == num rows 
            # keep this column index 
            greater_col.append(np.vstack(matrix[:,col]))

    if len(greater_col) > 0:
        return np.hstack(greater_col)
    else: 
        return np.array([])


# ---------------------------------------------------------------------
# QUESTION 6
# ---------------------------------------------------------------------


def filter_cutoff_np(matrix, cutoff):
    mean_arr = matrix.mean(axis=0)
    greater_col = mean_arr > cutoff
    return matrix[:, greater_col]


# ---------------------------------------------------------------------
# QUESTION 7
# ---------------------------------------------------------------------


def growth_rates(A):
    # our output array should have length == len(A)-1 
    # (final - initial)/initial
    # final == ith + 1 (starts at 1 and goes to the end)
    # initial == ith (initial starts at 0 but doesnt go to the end)

    growth_arr = (A[1:] - A[:-1])/A[:-1]
    print(len(A))
    print(len(growth_arr))
    return np.round(growth_arr, 2)

def with_leftover(A):
    num_stocks_bought = np.floor(20/A)
    amount_leftover = 20 - num_stocks_bought * A
    leftover_cumsum = np.cumsum(amount_leftover)
    
    if np.count_nonzero(leftover_cumsum > A):
        first_day = np.where(leftover_cumsum > A)
        return first_day[0][0]
    else:
        return -1


# ---------------------------------------------------------------------
# QUESTION 8
# ---------------------------------------------------------------------


def salary_stats(salary):
    num_players = salary.groupby('Player').count().shape[0]
    num_teams = salary.groupby('Team').count().shape[0]
    total_salary = salary['Salary'].sum()
    highest_salary = salary.groupby('Player')['Salary'].sum().sort_values(ascending=False).index[0] 
    avg_los = round((salary.groupby('Team')['Salary'].sum()/salary.groupby('Team')['Player'].count()).loc['Los Angeles Lakers'],2)
    fifth_lowest = (salary.groupby('Player')[['Team','Salary']].sum().sort_values('Salary',ascending=True).index[4] + ', ' + 
                    salary.groupby('Player')[['Team','Salary']].sum().sort_values('Salary',ascending=True)['Team'].iloc[4])
    
    # suffix are at the last element thus our 'Player' string goes [first, last, suffix] where last will always be index 1
    # majority of the time we will have duplicates
    # but if our sample is small (not representative) and replace=False we may have all unique last names 
    uniq_last = salary['Player'].str.split(' ').str[1].unique()
    duplicates = 0 
    if len(uniq_last) == salary.shape[0]:
        duplicates = False 
    else:
        duplicates = True
    
    team_name = salary.set_index('Player').loc[highest_salary]['Team']
    total_highest = salary.groupby('Team')['Salary'].sum().loc[team_name]

    data = [num_players,num_teams,total_salary,highest_salary,avg_los,fifth_lowest,duplicates,total_highest]
    index_ = ['num_players','num_teams','total_salary','highest_salary','avg_los','fifth_lowest','duplicates','total_highest']
    new_series = pd.Series(data, index=index_)
    return new_series


# ---------------------------------------------------------------------
# QUESTION 9
# ---------------------------------------------------------------------


def parse_malformed(fp):
    with open('data/malformed.csv','r') as file:
        df_rows = []

        for line in file:
            values = line.strip().split(',')
            geo = ','.join(values[4:])
            full_row =  values[:4] + [geo]
            df_rows.append(full_row)

    df = pd.DataFrame(data=df_rows[1:],index=None,columns=df_rows[0])

    df = df.apply(lambda col:col.str.replace('"','') if col.name=='geo' else col.str.replace('"','').str.replace(',',''))
    df['weight'] = pd.to_numeric(df['weight'])
    df['height'] = pd.to_numeric(df['height'])
    return df
