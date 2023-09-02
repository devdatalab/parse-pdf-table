from layout import tableDetection
#from ocr import ocrText
from column import columnDetection
from row import rowDetection
from sys import argv
import fitz
from PIL import Image
import pandas as pd
import pdb
import re

# --------------- #
# 1 Document Prep #
# --------------- #

# open the desired PDF page with fitz
doc = fitz.open(pdf_filename)

# get page object for pixmap
page = doc[doc_num]

# get all the words and their coordinates from the page
# words is a list of items (x0, y0, x1, y1, "word", block_no, line_no, word_no)
words = page.get_text("words")

# ------------------------------------------------------------------ #
# 2 Layout Detection: (Tool: ML Object Detection - Layout Parser) #
# ------------------------------------------------------------------ #

# set the options for LayoutParser, if we are using it
lp_options = {'modelType': 'tableBank', 'verbose': True}

# call our layout parser function wrapper
table_coords = get_table_coordinates_lp(page, pdf_filename, lp_options)

# alternatively, call our own function that finds the table coordinates
# this is probably project-specific and depends on the PDF layouts
table_coords = get_table_coordinates_custom(page, pdf_filename)

# for now, assume that only one table is ever found
if len(table_coords) > 1:
    print('So far, no implementation for multiple tables in a file')
    sys.exit()

# get all the PDF words inside the bounding box, and put them in a dataframe
df_words = pd.DataFrame([w for w in words if fitz.Rect(w[:4]).intersects(fitz.Rect(table_coords["x0"], table_coords["y0"], table_coords["x1"], table_coords["y1"]))])

# define the column names for the dataframe (there's probably a more concise way passing in a list)
column_names = ['x0', 'y0', 'x1', 'y1', 'text', 'block_no', 'line_no', 'word_no']
df_words = df_words.rename(columns=dict(enumerate(column_names)))

# ------------------------------------------------------------------ #
# 2.5 Custom steps to extract specific features from a page          #
# ------------------------------------------------------------------ #

# Here, we might have project-specific code, e.g. to grab a subdistrict name
# from a set of known coordinates on the page. This can be added to the output
# dataframe in the final section.


# ------------------------------------------------------------------------------------ #
# 3 Column Detection: (Tool: Heirarchical Agglomerative Clustering - Sci Kit Learn) #
# ------------------------------------------------------------------------------------ #

# set the agglomerative clustering options
col_detect_options = {'dist_thresh': 50, 'linkage_type': 'average'}

# Run column detection function. 
# This should run on df_words, which limits it to the inside of the table_coords bounding box
# This should return a dataframe with a column number for each word
# Do we need to return any other metadata?
# note: ColumnDetection() seems to be doing some project-specific stuff,
#       if so, we need to take all of this out and do it through the options dictionary.
df_columns = detect_columns(df_words, col_detect_options)

# merge the columns to the df_words dataframe
df_words.merge(...)

# ------------------------------------------------------------------------------------ #
# 3 Row Detection: (Tool: Naive Bayes - Python) #
# ------------------------------------------------------------------------------------ #

# set the row detecting parameters
row_detect_options = {'...':'...'}

# run the row detection algorithm
df_rows = detect_rows(df_words, row_detect_options)

# merge the row numbers to the df_words dataframe
df_words.merge(...)

# ------------------------------------------------------------------------------------ #
# 4 Save the result as a CSV                                                           #
# ------------------------------------------------------------------------------------ #

# group all the text in each cell (i.e. row/column pair) into one string,
#       and make a long dataframe from it
final_text = df.groupby(['col', 'row'])['text'].apply(' '.join).reset_index()

# reshape the dataframe to a simple table structure
out_df = final_text.pivot(columns='col', index='row', values='text')

# set the filename column
# [note: in some cases, this might be a string from the page, like a subdistrict name,
#        which would be detected through custom code above
out_df['pdf_filename'] = pdf_filename

# write it to a CSV
filepath = pdf_filename + '.csv'
out_df.to_csv(filepath)


