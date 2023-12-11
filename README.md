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

### Testing code

### Usage within a program

```
import pdfparser as parser
import pdb


out_df = parser.parse_pdf_table(pdf_str="~/iec/pc01/district_handbooks/DH_33_2001_KKU.pdf",
                                outfile = "/scratch/kjha/out.csv", page_no=278, dist_thresh=None, num_columns=7)
```

To parse a pdf, you need to specify a distance threshold `dist_thresh`, between 10 and 100 seems to work in many cases or specify the number of columns expected in the table `num_columns` but not both. One of these should always be none. For the use case this was designed on; district handbooks for the pc handbooks, specifying the number of columns works best, since we extract appendix tables that have the same number of columns across pages. 

Additional options to the function let you specify a bounding box within which to search for the table, which could be useful if this is used in a pipeline that uses layoutparser to detect tables in a pdf. 

In case you want to use this with a table without a text layer, use [ocrmypdf](https://ocrmypdf.readthedocs.io/en/v15.3.1/cookbook.html) locally on the docs to generate a text layer and then use ddlpdfparser on them.

