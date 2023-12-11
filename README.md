### The big idea

We read in tables from PDFs that have an underlying text layer for these tables.

## Quick start

Create a conda environment from the included config file:
```
conda env create -f parse-pdf-table.yml
conda activate parse-pdf-table
```
Navigate to the github directory, which on polaris looks usually like:
```
~/ddl/parse-pdf-table
```

Install the `pdfparser` library
```
pip install --editable .
```

### Minimum example

```
import pdfparser as parser
import pdb


out_df = parser.parse_pdf_table(pdf_str="~/iec/pc01/district_handbooks/DH_33_2001_KKU.pdf",
                                outfile = "/scratch/kjha/out.csv", page_no=278, dist_thresh=None, num_columns=7)
```

To parse a pdf, you need to specify a distance threshold `dist_thresh`, between 10 and 100 seems to work in many cases or specify the number of columns expected in the table `num_columns` but not both. One of these should always be none. For the use case this was designed on; district handbooks for the pc handbooks, specifying the number of columns works best, since we extract appendix tables that have the same number of columns across pages. 

Additional options to the function let you specify a bounding box within which to search for the table, which could be useful if this is used in a pipeline that uses layoutparser to detect tables in a pdf. 

In case you want to use this with a table without a text layer, use [ocrmypdf](https://ocrmypdf.readthedocs.io/en/v15.3.1/cookbook.html) locally on the docs to generate a text layer and then use ddlpdfparser on them.

### A slightly more involved example: Reading in a table from PC01 district pdf for Nagaland's Mokokchung district

```
  1 # load packages
  2 from ddlpy.utils.constants import *
  3 from pathlib import Path
  4 from text_eb_processing import longest_run_of_integers, find_appendix_pages
  5 import ddlpdfparser as parser
  6 import pdb
  7
  8
  9 ###################
 10 # Minimum Example #
 11 ###################
 12 pc01_hb = IEC / "pc01/district_handbooks"
 13 pdf_path = pc01_hb / "DH_13_2001_MOK.pdf"
 14
 15 list_matches = find_appendix_pages(pdf_path)
 16 list_appendix = longest_run_of_integers(list_matches)
 17 pdb.set_trace()
 18 # KJ: Change page number
 19 out_df = parser.parse_pdf_table(pdf_str=pdf_path, page_no=178, outfile = f"/scratch/kjha/out_178.csv",
 20                                 dist_thresh=None, num_columns=7, key_col=4)
 21
```
