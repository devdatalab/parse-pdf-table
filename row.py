import pandas as pd
import numpy as np
import pdb

# check if all characters of cell are digits
def contains_number(item):
    item = str(item)
    return item.isdigit()

# check if item is not a number
def is_nan(item):
    item = str(item)
    return item == "nan"

# given a dataframe of words and coordinates, assign each word to a row number
def rowDetection(df, df_row_input):

    # PN: note that df and df_row_input BOTH have column numbers, which mean different things.
    #     df is the correct column ordering, while df_row_input's columns are over-split.
    #     Otherwise, these two input dataframes are the same (!!)
    
    # initialize some values we will use later
    perc_num = 0
    max_num = 0
    max_col = 0
    
    # assign the first column as the key
    # PN: Note, in other iterations, we assigned the first string column as a key...
    key_col = 0

    # --------------------------------------- #
    # create a row dataframe from the row key #
    # --------------------------------------- #

    # GET A DATAFRAME WITH JUST THE ENTRIES IN THE VERY FIRST COLUMN (the "key" column)
    
    # create a dataframe with just information about the rows

    # get the key column from df_row_input
    df_row_key_int = df_row_input[['y0', 'y1', 'x1', 'text', 'col']]

    # subset the row dataframe to just include the key column. Copy it, so we can manipulate later.
    df_row_key = df_row_key_int[df_row_key_int['col'] == key_col].copy()

    # assign sequential row numbers to these fields.
    df_row_key.reset_index(inplace=True, drop=True)
    df_row_key['row'] = df_row_key.index

    # get the y midpoint for each row
    df_row_key['ymid'] = np.mean(df_row_key[['y0', 'y1']], axis=1)
    
    # For each entry in the key, make an array with all integer y values (e.g. [5, 6, 7, 8, 9]
    df_row_key["rowarray"] = df_row_key.apply(lambda row: np.arange(np.round(row["y0"]), np.round(row["y1"]+1)), axis=1)

    # sort the input dataframe by column, and then by y value.
    # PN: It's mainly important for words to be sorted by y0, as we use Y information on the last word assigned
    #     to handle skew.
    df = df.sort_values(by=["col", "y0"]).reset_index(drop=True)

    
    # ----------- #
    # This function does the Bayesian updating of where (in terms of Y) we expect entries in a given row to be found.
    # The idea is if we see an entry shifted slightly down, we expect future entries to also be shifted down.
    def update_row(row, df_row_key, theta, rowind):
        df_row_key.at[rowind[0], 'ymid'] = theta * df_row_key.loc[rowind[0], "ymid"] + ((1-theta) * np.mean(row[['y0', 'y1']]))
        df_row_key.at[rowind[0], 'y0'] = theta * df_row_key.loc[rowind[0], "y0"] + ((1-theta) * row['y0'])
        df_row_key.at[rowind[0], 'y1'] = theta * df_row_key.loc[rowind[0], "y1"] + ((1-theta) * row['y1'])
        df_row_key.at[rowind[0], 'rowarray'] = np.arange(np.round(df_row_key.loc[rowind[0], "y0"]), np.round(df_row_key.loc[rowind[0], "y1"]+1))
        return df_row_key
    
    # ----------- #
    # This function takes a row from the main dataframe, and matches it to a row from df_row_key.
    # It also updates df_row_key with information from that row --- each row affects where we
    # expect future entries in this row to appear.
    # ----------- #
    def match_to_row(row, df_row_key, theta):
        """
        Compare the y coords for each piece of text data
        to the coords determined for the rows. Determine the
        percentage overlap with this piece of text and every row,
        then pick the maximum overlap.
        """

        # create the array for the width of the piece of text data
        txtarray = np.arange(np.round(row["y0"]), np.round(row["y1"]))

        # we create two overlap measures. 1: (overlap y range) / (word height)
        #                                 2: (overlap y range) / (row height)
        # PN: change this to just do the math operation instead of using these discrete arrays
        
        # calculate the overlap between the txtarray and each row wrt the txt array
        df_row_key["overlap"] = df_row_key["rowarray"].apply(
            lambda y: len(np.intersect1d(y, txtarray)) / len(txtarray)
        )
        # calculate the overlap between the txtarray and each row wrt each row
        df_row_key["overlap2"] = df_row_key["rowarray"].apply(
            lambda y: len(np.intersect1d(y, txtarray)) / len(y)
        )
        
        # find the index with maximal overlap
        rowind = df_row_key.loc[df_row_key["overlap"] == df_row_key["overlap"].max()].index
        
        # if there are multiple overlaps at the max value, break ties with the 2nd overlap measure
        if len(rowind) > 1:
            rowind = df_row_key[df_row_key["overlap2"] == df_row_key.loc[rowind, "overlap2"].max()].index

        # isolate the row from the selected index
        # [PN: note this is arbitrarily picking a row if there is a list, but maybe this is fine.]
        r = df_row_key.loc[rowind[0], "row"]
        
        # if the sum is 0, then there is no overlap with any row
        # and the text data falls outside the row
        if df_row_key["overlap"].sum() == 0:
            r = np.nan

        # otherwise, update the df_row_key with the position of this entry --- it gives us more information
        #   about where this row tends to be found. And return the assigned row to the calling function.
        else:
            df_row_key = update_row(row, df_row_key, theta, rowind)
        return r
    
    # run the row assignment algorithm on each row,
    # PN: what is the meaning of starting theta=0.9?
    df['row'] = df.apply(lambda row: match_to_row(row, df_row_key, .9), axis=1)
    df = df[['y0', 'x1', 'row', 'text']]

    return df


    # PN: This whole block gets skipped in the 2001 census, since one.pdf
    #     is classified as RHS. Looks like we needed this for the 1951 census
    #     parse, so we should keep it until we understand it. This is the kind
    #     of code that would go in the wrapper function for the canals-specific
    #     analysis.
    # 
    # if page_type == "LHS":
    #     sorted_cols = df_row_input['col'].unique()
    #     sorted_cols = sorted_cols.tolist()
    #     while None in sorted_cols:
    #         sorted_cols.remove(None)
    #     sorted_cols.sort()
    #     # for every column in the dataframe
    #     for col in sorted_cols:
    #         page_df = df_row_input[df_row_input['col'] == col]
    #         # if it's a long page
    #         if len(page_df) > 30:
    #             # if the column that is 90% #s hasn't been hit
    #             if perc_num < 50:
    #                 # get a T/F array of whether all chars of value are digits
    #                 perc_num = page_df['text'].astype(str).apply(contains_number)
    #                 # get a T/F array of whether value is nan or is not
    #                 perc_nan = page_df['text'].astype(str).apply(is_nan)
    #                 # get the percent of the non-nan values in a column that are numbers
    #                 perc_num = (perc_num.sum() / (perc_nan.count() - perc_nan.sum())) * 100
    #                 key_col = col
    #                 if perc_num > max_num:
    #                     max_num = perc_num
    #                     max_col = col
