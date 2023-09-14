# file for helper functions. Might break up if there are many row/column specific detection functions used

###########
# Imports #
###########
import pandas as pd
import numpy as np
import pdb

#####################################
# Column Detection Helper Functions #
#####################################
def within_bounding_box(df:pd.DataFrame, bounding_box:list) -> pd.DataFrame:
    """
    This function checks if the DataFrame provided has coordinates that lie within the bounding box provided
    """
    # Check if dataframe has x0, x1, y0 and y1 coordinates for each element/word. throw an error otherwise

    def filter_row(row, bounding_box):
        """
        Small function to filter rows
        """
        # KJ: Add error handling block to raise an error if bounding box is empty
        min_x, min_y, width, height = bounding_box
        max_x = min_x + width
        max_y = min_y + height
        if (min_x <= row['x0'] <= max_x and  min_x <= row['x1'] <= max_x and min_y <= row['y0'] <= max_y and min_y <= row['y0'] <= max_y):
            return True
        else:
            return False

    filtered_df = df[df.apply(filter_row,args=(bounding_box,), axis=1)].reset_index()

    return filtered_df
        

######################################
# Row Detection Helper Functions     #
######################################


# ----------- #
# This function does the Bayesian updating of where (in terms of Y) we expect entries in a given row to be found.
# The idea is if we see an entry shifted slightly down, we expect future entries to also be shifted down.
def update_row_key(row, df_row_key, theta, row_index):

    # for each row characteristic R, we set the value to [ theta * R + (1 - theta) * (this row) ]
    df_row_key.at[row_index, 'y0']   = theta * df_row_key.loc[row_index, "y0_key"]   + ((1 - theta) * row['y0'])
    df_row_key.at[row_index, 'y1']   = theta * df_row_key.loc[row_index, "y1_key"]   + ((1 - theta) * row['y1'])

    return df_row_key

# ----------- #
# This function takes a row from the main dataframe, and matches it to a row from df_row_key.
# It also updates df_row_key with information from that row --- each row affects where we
# expect future entries in this row to appear.
# ----------- #
def match_to_row(row:pd.Series, df_row_key:pd.DataFrame, theta:float=1):
    """
    Compare the y coords for each piece of text data
    to the coords determined for the rows. Determine the
    percentage overlap with this piece of text and every row,
    then pick the maximum overlap.
    """
    
    # get overlap between dataframe row and key
    df_row_key["overlap"] = get_overlap_key_row(row, df_row_key)
    
    # find the index with maximal overlap    
    row_index = df_row_key.loc[df_row_key["overlap"] == df_row_key["overlap"].max()].index
    
    # at this point, the row index can still be a list, so we just pick the first. 
    # KJ: think about making this better later. Seems to work well enough for now.
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
    
def get_overlap_key_row(row:pd.Series, df_row_key:pd.DataFrame):
    """
    Helper function to calculate overlap between a row (of a dataframe passed as a series object) and the df_row_key, which has positional information
    from the key column we use to separate rows. 
    """
    overlap = np.maximum(0,(np.minimum(row["y1"], df_row_key["y1_key"]) - np.maximum(row["y0"], df_row_key["y0_key"])))
    row_length = row["y1"] - row["y0"]
    overlap = overlap/row_length
    return overlap
