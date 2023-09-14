import ddlpdfparser as parser
import pdb

  
# bounding_box = [50,150,500,600] 
# this is from manual inspection of DH_24_2001_BHN.pdf

    # KJ: Oversplit method doesn't always work on all pages. 
    # df_row_input = detect_columns(words_df, dist_thresh=8, linkage_type="average")
    # Fails for parse_pdf_table("~/iec/pc01/district_handbooks/DH_33_2001_KKU.pdf", 278)
    # df_row_input = df_columns, with key_column set to 0, works for this. 

    
    # KJ: This distance threshold parameter seems to be determined manually. And is rather brittle to changes.
    # Moreover, it's hard to know beforehand, especially if there are many tables without varying layouts if this will
    # work for all of them. 

out_df = parser.parse_pdf_table(pdf_str="~/iec/pc01/district_handbooks/DH_33_2001_KKU.pdf",
                                outfile = "/scratch/kjha/out.csv", page_no=278, dist_thresh=None, num_columns=7)
