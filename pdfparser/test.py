import pdfparser as parser


parser.parse_pdf_table("~/iec/pc01/district_handbooks/DH_33_2001_KKU.pdf", 278, dist_thresh=None, num_columns=7)
