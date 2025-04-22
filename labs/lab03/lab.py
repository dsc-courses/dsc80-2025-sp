# lab.py


import os
import io
from pathlib import Path
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def read_linkedin_survey(dirname):
    # open file path to create a file object 
    surveys_fp = Path(dirname)

    surveys_cvs = []
    try:
        for file in surveys_fp.iterdir():
            surveys_cvs.append(file)
    except FileNotFoundError:
        raise

    dfs_ls = []
    for file in surveys_cvs:
        new_df = pd.read_csv(file)
        dfs_ls.append(new_df)

    ordered_col = ['first name', 'last name', 'current company', 'job title', 'email', 'university']

    combined_df = pd.DataFrame(columns=ordered_col)
    for df in dfs_ls:
        df.columns = df.columns.str.replace('_',' ').str.lower()
        combined_df = pd.concat([combined_df,df],ignore_index=True)
    
    return combined_df


def com_stats(df):
    ohio_filtered_df = df[df['university'].str.contains('Ohio',na=False)]
    prop_ohio_programmer = ohio_filtered_df[ohio_filtered_df['job title'].str.contains('Programmer',na=False)].shape[0] / ohio_filtered_df.shape[0]

    # -----------------------------------------------------------------------------------------------------------------------------

    num_eng_jobs = len(df[df['job title'].str.endswith('Engineer',na=False)]['job title'].unique())

    # -----------------------------------------------------------------------------------------------------------------------------

    longest_job_name_df = df.copy().dropna()
    max_index = longest_job_name_df['job title'].str.len().idxmax()
    longest_job_title = df.iloc[max_index]['job title']

    # -----------------------------------------------------------------------------------------------------------------------------

    find_manager_df = df.copy()
    find_manager_df['job title'] = find_manager_df['job title'].str.lower()
    num_managers = find_manager_df['job title'].str.contains('manager').sum()

    return [prop_ohio_programmer,num_eng_jobs,longest_job_title,num_managers]


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------


def read_student_surveys(dirname):
    file_fp = Path(dirname)

    all_questions = []
    for file in file_fp.iterdir():
        all_questions.append(file)

    questions_dfs_ls = []
    for file in all_questions:
        new_df = pd.read_csv(file)
        questions_dfs_ls.append(new_df)

    merged_questions_df = pd.DataFrame({'id':np.arange(1,1001)})
    for df in questions_dfs_ls:
        merged_questions_df = merged_questions_df.merge(df, on='id')

    merged_questions_df = merged_questions_df.set_index('id')

    return merged_questions_df


def check_credit(df):
    # clean col values 
    invalid_values = ["-", "Unknown", "", "None", "N/A", "(no genres listed)"]
    df = df.replace(invalid_values,np.nan)

    # (num of null entries for each question) / (total num of entries)
    # check wherever this is < 0.10 because we want at least 90% of students to answer so for there to be 10% null values allowed 
    num_entire_class_earns = ((df.set_index('name').isna().sum() / df.shape[0]) < 0.10).sum()

    if num_entire_class_earns > 2: 
        num_entire_class_earns = 2 

    swapped_df = df.copy()
    swapped_df = swapped_df.set_index('name')
    num_questions = len(swapped_df.columns)

    # (num of null entries per student) / (num of total questions) == prop of null entries per student 
    # want this to be less than 50%
    students_plus_five = (swapped_df.T.isna().sum() / num_questions) < 0.5

    swapped_df['recieves 5 EC'] = (students_plus_five.values) * 5 
    swapped_df['total ec'] = swapped_df['recieves 5 EC'].values + num_entire_class_earns

    ec_per_student = pd.DataFrame()
    ec_per_student.index = df.index
    ec_per_student['name'] = swapped_df.index
    ec_per_student['ec'] = swapped_df['total ec'].values

    return ec_per_student


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def most_popular_procedure(pets, procedure_history):
    ...

def pet_name_by_owner(owners, pets):
    ...


def total_cost_per_city(owners, pets, procedure_history, procedure_detail):
    ...


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------


def average_seller(sales):
    ...

def product_name(sales):
    ...

def count_product(sales):
    ...

def total_by_month(sales):
    ...
