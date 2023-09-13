# This has the row and column detection functions

from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from tabulate import tabulate
import pdb
from utils import within_bounding_box, match_to_row, update_row_key
import re

# given a dataframe from ocr'd text and AC parameters, assign each piece of text a column using HAC
# return a dataframe with x0, y0, and column information
def detect_columns(df:pd.DataFrame, dist_thresh:int=50, linkage_type:str="average",
                   num_columns:int=None, bounding_box:list=None, x_index:str="x1",
                   metric_type:str="manhattan"):
    """
    detect_columns uses Hierarchical Agglomerative Clustering(HAC) to determine columns in a table with 
    a text layer.

    Input arguments:
    df: a dataframe with positional information about words.
    dist_threshold: Input argument provided to HAC to cluster columns.
    linkage_type:
    num_columns: Should be None if dist_threshold is not None. Defaults to None.
    bounding_box: List of [min_x, min_y, max_x, max_y] coordinates that the words from dataframe must be within to be considered part of the table.
    x_index: Which x index of word to take into account when clustering. Can be x0, x1 or xmid
    metric_type: HAC parameter
    """
    # To DO: If neither dist_thresh nor num_columns passed, raise an error
    
    # filter words in the dataframe to be within the bounding box
    if bounding_box != None:
        df = within_bounding_box(df, bounding_box)
    
    # create the column dataframe which will be returned after column assignment
    # note the input dataframe has x0,x1, while the output only has x1, since x values
    # will be snapped to columns.
    #KJ: Can I add back 'x0' here safely without breaking anything??
    df_col = df[['x1', 'y0', 'y1', 'text']].copy()
    df_col['col'] = None

    # select the x coordinates from the df and convert them to correct format for HAC
    # have a placeholder value for columns, i.e. add a 0 to default coloumn value.
    xCoords = df_col.apply(lambda x: (x[x_index], 0), axis=1)
    xCoords = xCoords.values.tolist()

    # apply hierarchical agglomerative clustering to the coordinates
    clustering = AgglomerativeClustering(
        n_clusters = num_columns,
        metric = metric_type, 
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
        # KJ: Not sure what this does/which edge case this takes care of.
        if len(indexes) > min_cluster_size:
            
            # KJ: This should be passed as an argument.
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

# given a dataframe of words and coordinates, assign each word to a row number
def detect_rows(df:pd.DataFrame, df_row_key:pd.DataFrame, theta:int=1):
      
      # KJ: Note: df_row_key no longer has a confusing "col" column, it's only needed for selecting the desired row, 
      # which we do and then get rid of.
      # PN: It's mainly important for words to be sorted by y0, as we use Y information on the last word assigned
      # to handle skew.
      df = df.sort_values(by=["col", "y0"]).reset_index(drop=True)
      
      # run the row assignment algorithm on each row,
      # KJ: Theta parameter is for the bayesian updating of slanted tables, default set to 1. Can edit across projects.
      # Rename to bayes_theta so there's no doubt about this    
      df['row'] = df.apply(lambda row: match_to_row(row, df_row_key, theta=1), axis=1)
      df = df[['y0', 'x1', 'row', 'text']]

      return df

def get_key_row(words_df:pd.DataFrame, columns:pd.DataFrame, dist_thresh:int=None,linkage_type:str="average", 
                num_columns:int=None, key_col:int=0, oversplit_col=False):
            
            """
            This function gets the dataframe with row information for the column we want to snap all other rows to. 
            Think of this as the scaffold for row detection.
            """

            # KJ: If both dist_thresh and num_columns is None, raise an error:

            # By default, use the columns detected in detect_columns()
            if oversplit_col == False:
                    df_row_key = columns 

            # If oversplit defined, split the columns with a smaller distance threshold. 
            # KJ: Note, this 8 is a carryover from 1951 census, we may want to change this later on.
            if oversplit_col == True:
                    df_row_key = detect_columns(words_df, dist_thresh=8, linkage_type="average")
                    
            # get a dataframe with just the entries for the key_col column
            df_row_key = df_row_key.loc[df_row_key['col'] == key_col].drop(["col"], axis=1).reset_index(drop=True)

            # assign sequential row numbers to these fields.
            df_row_key['row'] = df_row_key.index 

            # pull in block, line and word numbers from the original dataframe
            # KJ: Don't think this does much here and so commenting out for now. 
            # df_row_key = pd.merge(words_df, df_row_key, on=['x1', 'y0', 'y1', 'text'])

            # rename columns to be "_key" for specific fields passed to them
            rename_pattern = re.compile('^([y][0-9])')
            df_row_key.columns = df_row_key.columns.map(lambda col_name: rename_pattern.sub('\\1_key', col_name))

            return df_row_key