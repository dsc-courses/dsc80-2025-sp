# lab.py


import os
import pandas as pd
import numpy as np
import requests
import bs4
import lxml


# ---------------------------------------------------------------------
# QUESTION 1
# ---------------------------------------------------------------------


def question1():
    """
    NOTE: You do NOT need to do anything with this function.
    The function for this question makes sure you
    have a correctly named HTML file in the right
    place. Note: This does NOT check if the supplementary files
    needed for your page are there!
    """
    # Don't change this function body!
    # No Python required; create the HTML file.
    return


# ---------------------------------------------------------------------
# QUESTION 2
# ---------------------------------------------------------------------



def extract_book_links(text):
    soup = bs4.BeautifulSoup(text, features='lxml')

    my_ls = []
    for article in soup.find_all('article',attrs={'class':'product_pod'}):
        book_url = article.find('a').get('href')

        book_price = article.find('p', attrs={'class':'price_color'}).text

        book_rating = article.find('p', attrs={'class':r'star-rating'}).get('class')[1]

        my_dict = {'url':book_url, 'price':book_price, 'rating':book_rating}
        my_ls.append(my_dict)

    book_links_df = pd.DataFrame(my_ls)
    book_links_df['price'] = book_links_df['price'].str.strip('Â£').astype(float)

    def convert_to_int(x):
        if x=='One':
            x = 1
        elif x=='Two':
            x = 2
        elif x=='Three':
            x = 3
        elif x=='Four':
            x = 4
        else:
            x = 5
        return x

    book_links_df['rating'] = book_links_df['rating'].apply(convert_to_int)
    book_links_df = book_links_df.query('price < 50 and rating >= 4')
    
    return book_links_df['url'].values.tolist()

def get_product_info(text, categories):
    soup2 = bs4.BeautifulSoup(text, features='lxml')

    book_cat = soup2.find('ul', attrs={'class':'breadcrumb'}).find_all('a')[-1].text
    if book_cat not in categories:
        return None

    book_title = soup2.find('div', attrs={'class':r'product_main'}).find('h1').text
    book_rating = soup2.find('p', attrs={'class':r'star-rating'}).get('class')[1]
    book_description = soup2.find('div', attrs={'id':'product_description'}).find_next_sibling('p').text

    col_labels = [x.text for x in soup2.find('table', attrs={'class':'table table-striped'}).find_all('th')]
    corr_col_values = [x.text for x in soup2.find('table', attrs={'class':'table table-striped'}).find_all('td')]

    my_dict = dict(zip(col_labels,corr_col_values))
    my_dict['Category'] = book_cat
    my_dict['Rating'] = book_rating
    my_dict['Description'] = book_description
    my_dict['Title'] = book_title
    
    return my_dict

def scrape_books(k, categories):
    combined_ls = []
    for i in range(k):
        # get requests book page 
        res = requests.get(f"http://books.toscrape.com/catalogue/page-{i+1}.html")

        # send to extract_book_links
        book_url = extract_book_links(res.text)
        
        # send url list to get_product_info
        for url in book_url:
            full_url = f"http://books.toscrape.com/catalogue/{url}"
            res_book = requests.get(full_url)
            my_dict = get_product_info(res_book.text, categories)

            # dont add None values because we will have issues adding it to Dataframe 
            if my_dict != None:
                combined_ls.append(my_dict)

    return pd.DataFrame(combined_ls)


# ---------------------------------------------------------------------
# QUESTION 3
# ---------------------------------------------------------------------


def stock_history(ticker, year, month):
    date_from = f"{year-1}-01-01"
    start_date = f"{year}-{month:02d}-1"
    end_date = f"{year}-{month+1:02d}-1"

    full_url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?from={date_from}&apikey=wz83furV3v1MWLGeQR9H8CO8lRaGuIPr"
    )
    res = requests.get(full_url)
    ticker_df = pd.DataFrame(res.json()['historical'])
    ticker_df['date'] = pd.to_datetime(ticker_df['date'])

    date_range = pd.date_range(start=start_date, end=end_date, inclusive='left')
    filtered_df = ticker_df[ticker_df['date'].isin(date_range)]

    return filtered_df

def stock_stats(history):
    open_start = history['open'].iloc[-1]
    close_end = history['close'].iloc[0]
    percent_change = ((close_end - open_start)/open_start)*100

    trans_vol_ser = ((history['high']+history['low'])/2)*(history['volume']/1_000_000_000)
    total_trans_vol = trans_vol_ser.sum()

    str_total_trans_vol = f"{total_trans_vol:.2f}B"
    if percent_change > 0:
        str_percent_change = f"+{percent_change:.2f}%"
        return (str_percent_change,str_total_trans_vol)
    else:
        str_percent_change = f"-{percent_change:.2f}"
        return (str_percent_change,str_total_trans_vol)


# ---------------------------------------------------------------------
# QUESTION 4
# ---------------------------------------------------------------------

def iterative_func(starting_ids):
    # pass a list of the kid's ids 
    pending = list(reversed(starting_ids))      # must do this for depth first not BFS 
    visited = set()
    next_up = []

    while pending:
        # while pending is not empty

        # grab first element in pending
        comment_id = pending.pop()

        if comment_id in visited:
            continue
        visited.add(comment_id)

        # format: https://hacker-news.firebaseio.com/v0/item/{...}.json?print=pretty
        res_comment = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{comment_id}.json?print=pretty")
        comment = res_comment.json()

        if comment is None:
            continue

        if 'kids' in comment:
            pending.extend(reversed(comment['kids']))

        if 'dead' in comment:
            continue 

        next_up.append(comment)
    
    return next_up

def get_comments(storyid):
    res_article = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{storyid}.json")
    parent = res_article.json()

    all_comments = iterative_func(parent['kids'])

    comments_df = pd.DataFrame(all_comments)
    comments_df['time'] = pd.to_datetime(comments_df['time'], unit='s')
    comments_df = comments_df.drop(columns=['type','kids'])

    return comments_df
