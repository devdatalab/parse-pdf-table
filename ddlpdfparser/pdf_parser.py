# from layout import tableDetection
#from ocr import ocrText
from table_detection import detect_columns,detect_rows, get_key_row
from sys import argv
import fitz
from PIL import Image
import pandas as pd
import pdb
import re
from pathlib import Path
from utils import within_bounding_box

def parse_pdf_table(pdf_str:str, page_no:int, outfile:str=None, key_col:int=0, oversplit=False, num_columns=None, column_linkage_type="average", 
                    col_x_index:str="x1", dist_thresh:int=50, bounding_box:list=None, metric_type="manhattan", theta:float=0.9):
    """
    This function takes the pdf_path and page number of the pdf as an input, reads in the table 
    on the page. Assumes entire page is the table unless a bounding_box is specified. 
    
    """
    ##################
    # Error checking #
    ##################
    # Only one of dist_thresh or num_columns should be provided as HAC arguments
    if (dist_thresh != None and num_columns != None):
        raise ValueError("Either dist_thresh or num_columns must be none")
    
    #################################################################
    # To Do: Add layout parser block here to detect table on page # #
    #################################################################

    #########################################
    # Use fitz to read in words on the page #
    #########################################

    # read in the pdf string as a path variable so it can be used as an object everywhere else, expanduser to ensure any ~ are resolved correctly
    pdf_path = Path(pdf_str).expanduser()

    pdf = fitz.open(pdf_path)

    pdf_page = pdf[page_no]

    words = pdf_page.get_text("words")

    words_df = pd.DataFrame(words)

    # set column names
    words_df = words_df.rename(
        columns={
            0: 'x0',
            1: 'y0',
            2: 'x1',
            3: 'y1',
            4: 'text',
            5: 'block_no',
            6: 'line_no',
            7: 'word_no'
        }
    )


    #####################################################################################
    # Column Detection: (Tool: Heirarchical Agglomerative Clustering - Sci Kit Learn) # #
    #####################################################################################

    # Set parameters for column detection
    # the column function takes an ocr dataframe and identifies clusters based on their relative distance from one another
    # the dataframe returned is a key: item dataframe where key is column number and item is text within the current column
    df_columns = detect_columns(df=words_df, dist_thresh=dist_thresh, linkage_type=column_linkage_type, 
                                num_columns=num_columns, bounding_box=bounding_box, x_index=col_x_index, metric_type=metric_type)

    # merge words_df with column info
    words_df = pd.merge(words_df, df_columns, on=['x1', 'y0', 'y1', 'text'])

    ###################
    # Row Detection # #
    ###################
    # arrange words dataframe by y0 so we can order on rows. 
    words_df.sort_values(by=['y0'], ascending=[True], inplace=True)
    
    # Get's a master row key, from a key column (that has all row entries), to which we snap the other
    # rows (of other columns) to.
    df_row_key = get_key_row(words_df=words_df, columns=df_columns, num_columns = num_columns, key_col=key_col, oversplit_col=oversplit)

    # run the row detection algorithm to snap all rows from all columns to the master row key in df_row_key
    df_rows = detect_rows(words_df, df_row_key, theta=theta)

    # merge the row numbers to the original dataframe
    words_df = pd.merge(words_df, df_rows, on=['x1', 'y0','text'])

    # Sort by x and y coodinates
    # KJ: Need to check if this causes errors across PDFs. 
    words_df.sort_values(by=['y0', 'x0'], ascending=[True, True], inplace=True)
    
    # reshape the dataframe to align values by row and column
    final_text = words_df.groupby(['col', 'row'])['text'].apply(' '.join).reset_index()
    out_df = final_text.pivot(columns='col', index='row', values='text')

    # write it to a CSV
    if outfile != None: 
        outfile = Path(outfile).expanduser()
        out_df.to_csv(outfile)
        print(f"Table written to {outfile}")
        
    return out_df

#if __name__ == '__main__':
#    #print(parse_pdf_table("~/iec/pc01/district_handbooks/DH_33_2001_KKU.pdf", 278, dist_thresh=None, num_columns=7))
#    #print(parse_pdf_table("~/iec/pc01/district_handbooks/DH_24_2001_BHN.pdf", 490))
#    print(parse_pdf_table("~/iec/pc01/district_handbooks/DH_24_2001_BHN.pdf", 490, dist_thresh=None, num_columns=7))
