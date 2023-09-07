# from layout import tableDetection
#from ocr import ocrText
from table_detection import detect_columns,detect_rows
from sys import argv
import fitz
from PIL import Image
import pandas as pd
import pdb
import re
from pathlib import Path


def parse_pdf_table(pdf_str:str, page_no:int):
    """
    This function takes the pdf_path and page number of the pdf as an input, reads in the table 
    on the page (currently assumes the full page is the table). 
    
    """

    ########################################################
    # Add layout parser block here to detect table on page #
    ########################################################

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

    column_dist_thresh = 50
    linkage_type = "average"
    # PN: get the full word list into a dataframe
    
    # the column function takes an ocr dataframe and identifies clusters based on their relative distance from one another
    # the dataframe returned is a key: item dataframe where key is column number and item is OCR'd text within the current column
    df_columns = detect_columns(words_df, column_dist_thresh, linkage_type)


    ###############################################
    # Row Detection: (Tool: Naive Bayes - Python) #
    ###############################################

    # get a column assignment that is more accurate for key purposes
    # PN: We get bad results if we swap these two lines. This is bad, sort order shouldn't matter!
    #     The internal functions should figure out the best sort order to use (or allow it to be specified
    #     in a parameter.)
    df_row_input = detect_columns(words_df, 8, linkage_type)
    words_df.sort_values(by=['y0'], ascending=[True], inplace=True)

    # pull in block, line and word numbers from the original dataframe
    df_row_input = pd.merge(words_df, df_row_input, on=['x1', 'y0', 'y1', 'text'])
    
    
    # merge the CORRECT column classification into the primary dataset
    # PN: Note this is pretty confusing b/c we have two column classifications, both called 'col', 1 in words_df_columns and 1 in df_row_input
    words_df = pd.merge(words_df, df_columns, on=['x1', 'y0', 'y1', 'text'])
    
    # run the row detection algorithm
    # PN: scarily, both of these have a 'col' column, but based on a different run of the column detection algorithm!
    df_rows = detect_rows(words_df, df_row_input)

    # merge the row numbers to the original dataframe
    words_df = pd.merge(words_df, df_rows, on=['x1', 'y0','text'])

    # PN: Ellie's code suggested that sort order might matter here -- I'm not sure why, but leaving the commented lines
    #     # sort all values by y0, this should be sorting by rows
    #     df.sort_values(by=['y0'], ascending=[True], inplace=True)
    #     # IMPORTANT: added this for the new census example... if there are issues with 1951 now this is the first thing to comment out
    #     df.sort_values(by=['x0'], ascending=[True], inplace=True)
    # PN: I replace with a sort on row, then column
    #     This should only affect the word order within cells --- the best approach depends on whether there are more X or Y errors. 
    #     row, then column, fits normal reading, but may be different for some PDFs
    words_df.sort_values(by=['y0', 'x0'], ascending=[True, True], inplace=True)
    
    # combine text strings that appear in the same cells
    final_text = words_df.groupby(['col', 'row'])['text'].apply(' '.join).reset_index()

    # reshape the dataframe to align values by row and column
    out_df = final_text.pivot(columns='col', index='row', values='text')

    # write it to a CSV
    filepath = 'out/final_output{}.csv'.format(page_no)
    out_df.to_csv(filepath)
    
    print("done")

    return None

if __name__ == '__main__':
    print(parse_pdf_table("~/ddl/parse-pdf-table/one.pdf", 483))
