import ddlpdfparser as parser
import pdb


out_df = parser.parse_pdf_table(pdf_str="~/iec/pc01/district_handbooks/DH_33_2001_KKU.pdf",
                                outfile = "/scratch/kjha/out.csv", page_no=278, dist_thresh=None, num_columns=7)
