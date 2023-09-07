from sklearn.cluster import AgglomerativeClustering
import pandas as pd
import numpy as np
from tabulate import tabulate
import pdb

# given a dataframe from ocr'd text and AC parameters, assign each piece of text a column using HAC
# return a dataframe with x0, y0, and column information
def columnDetection(df, dist_thresh, linkage_type):

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
