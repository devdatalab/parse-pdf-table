# This has the row and column detection functions

from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from tabulate import tabulate
import pdb

# given a dataframe from ocr'd text and AC parameters, assign each piece of text a column using HAC
# return a dataframe with x0, y0, and column information
def detect_columns(df, dist_thresh, linkage_type):

    # create the column dataframe which will be returned after column assignment
    # note the input dataframe has x0,x1, while the output only has x1, since x values
    # will be snapped to columns.
    df_col = df[['x1', 'y0', 'y1', 'text']].copy()
    df_col['col'] = None

    # select the x coordinates from the df and convert them to correct format for HAC
    # have a placeholder value for columns, i.e. add a 0 to default coloumn value.
    xCoords = df_col.apply(lambda x: (x['x1'], 0), axis=1)
    xCoords = xCoords.values.tolist()
    
    # apply hierarchical agglomerative clustering to the coordinates
    clustering = AgglomerativeClustering(
        n_clusters = None,
        affinity = "manhattan", # manhattan
        linkage = linkage_type,
        distance_threshold = dist_thresh)
    clustering.fit(xCoords)
    
    # initialize our list of sorted clusters
    sorted_clusters = []
    min_cluster_size = 2
    
    # loop over all clusters
    for label in np.unique(clustering.labels_):
        
        # extract the indexes for the coordinates belonging to the
        # current cluster
        indexes = np.where(clustering.labels_ == label)[0]
        
        # verify that the cluster is sufficiently large
        if len(indexes) > min_cluster_size:
            
            # compute the average x-coordinate value of the cluster
            # note we use the original df here, not df_col, because df_col
            # does not have x0 --- df_col only has a single "snapped" x value.
            avg = np.average([df.loc[indexes,'x0']])

            # update the cluster list with the current label and the average x value
            sorted_clusters.append((label, avg))
            
    # sort the clusters by their average x-coordinate
    sorted_clusters.sort(key = lambda x: x[1])

    # loop over the clusters again, this time in sorted order, to add column numbers
    col_num = 0
    for (label, _) in sorted_clusters:
        
        # extract the indexes for the coordinates belonging to the current cluster
        indexes = np.where(clustering.labels_ == label)[0]

        # add this column number to the df, and increment
        df_col.loc[indexes, 'col'] = col_num
        col_num = col_num + 1
        
    # replace NaN values with empty strings
    df.fillna("", inplace=True)

    # return a dataframe with text, coordinates (with column-snapped x values), and column numbers
    return df_col


###########################
# Row Detection Functions #
###########################

# check if all characters of cell are digits
def contains_number(item):
    item = str(item)
    return item.isdigit()

# check if item is not a number
def is_nan(item):
    item = str(item)
    return item == "nan"

# given a dataframe of words and coordinates, assign each word to a row number
def detect_rows(df, df_row_input):
    
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
    def update_row_key(row, df_row_key, theta, row_index):

        # for each row characteristic R, we set the value to [ theta * R + (1 - theta) * (this row) ]
        # PN: note: we update ymid, but it never gets used again.
        df_row_key.at[row_index, 'ymid'] = theta * df_row_key.loc[row_index, "ymid"] + ((1 - theta) * np.mean(row[['y0', 'y1']]))
        df_row_key.at[row_index, 'y0']   = theta * df_row_key.loc[row_index, "y0"]   + ((1 - theta) * row['y0'])
        df_row_key.at[row_index, 'y1']   = theta * df_row_key.loc[row_index, "y1"]   + ((1 - theta) * row['y1'])

        # rebuild the row integer array based on the updated values
        df_row_key.at[row_index, 'rowarray'] = np.arange(np.round(df_row_key.loc[row_index, "y0"]), np.round(df_row_key.loc[row_index, "y1"] + 1))
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
        row_index = df_row_key.loc[df_row_key["overlap"] == df_row_key["overlap"].max()].index
        
        # if there are multiple overlaps at the max value, break ties with the 2nd overlap measure
        if len(row_index) > 1:
            row_index = df_row_key[df_row_key["overlap2"] == df_row_key.loc[row_index, "overlap2"].max()].index

        # at this point, the row index can still be a list, but if so, there is no further way to reconcile,
        # so we just pick the first
        row_index = row_index[0]
            
        # isolate the row from the selected index
        r = df_row_key.loc[row_index, "row"]
        
        # if the sum is 0, then there is no overlap with any row
        # and the text data falls outside the row
        if df_row_key["overlap"].sum() == 0:
            r = np.nan

        # otherwise, update the df_row_key with the position of this entry --- it gives us more information
        #   about where this row tends to be found. And return the assigned row to the calling function.
        else:
            df_row_key = update_row_key(row, df_row_key, theta, row_index)
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
