# 03.3_reader_nlp.py

# Starting from log-it (or in general log-country), extract new dates reading the XML

# Output: new file with case-id and date --> event_log_IT_BID-OPENING_method.csv


import pandas as pd
import os
from datetime import datetime
import glob
import PyPDF2

# NLP
import sparknlp
from sparknlp.annotator import DocumentAssembler, DateMatcher, RegexMatcher
from pyspark.sql.types import StringType
from pyspark.ml import Pipeline

# GLOBALS
data_dir = "data" # input / output data folder
guue_dir = "guue" # PDF folder
file_log = "event_log_IT.csv" # <-- INPUT
file_new_date_pdf = "event_log_IT_BID-OPENING_nlp_#.csv" # the output filename with new event
file_nlp_log = "spark_log_#.csv" # NLP task log

annotator_type = {'DateMatcher':1, 'RegexMatcher':2} # annotator type dict
annotator = annotator_type["RegexMatcher"] # <-- INPUT: choose the annotator

ted_url = "https://ted.europa.eu/TED" # to remove it from data

# FUNCTIONS
def date_to_utc(date, sep):
    """ Given a date and the separator, return it in yyyy-mm-dd """
    padding = "0"
    try:
        parts = date.split(sep)
        mm = str(parts[1].rjust(2, padding))
        dd = str(parts[0].rjust(2, padding))
        date_utc = parts[2]+"-"+mm+"-"+dd
        return date_utc
    except IndexError:
        return None
    

def spark_date(list_text, file_name, annotator_type):

    spark = sparknlp.start()

    print("Apache Spark version:", spark.version)

    documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")

    # .setRules(["\d{4}\/\d\d\/\d\d,date", "\s\d{2}\/\d\d\/\d\d,short_date"]) \
    
    # test 1
    if annotator_type == 1:

        date = DateMatcher() \
        .setInputCols("document") \
        .setOutputCol("date") \
        .setSourceLanguage("it")
        #.setOutputFormat("yyyy-MM-dd") \
        
        pipeline = Pipeline().setStages([
        documentAssembler,
        date
        ])
    

    # test 2
    if annotator_type == 2:
        regex_matcher = RegexMatcher()\
        .setRules(["\d{2}\/\d{2}\/\d{4}, date_1", "\d{2}\.\d{2}\.\d{4}, date_2,", "\d{2}\.\d\.\d{4}, date_3", "\d{1}\.\d{1}\.\d{4}, date_4", "\d{1}\/\d{1}\/\d{4}, date_5"]) \
        .setDelimiter(",") \
        .setInputCols("document") \
        .setOutputCol("date") \
        .setStrategy("MATCH_ALL")

        pipeline = Pipeline().setStages([
        documentAssembler,
        regex_matcher
        ])
    
    # perform the task
    spark_df = spark.createDataFrame(list_text, StringType()).toDF("text")
    result = pipeline.fit(spark_df).transform(spark_df)
    result.selectExpr("text","date.result as date").show(truncate=False)
    # print(result) # debug
    result_df = result.toPandas()
    result_df.insert(0, 'file_name', file_name) # add the file_name to the result df
    # print(result_df.info()) # debug
    result_df.to_csv(file_nlp_log, sep = ";", index = False, header=False, mode="a") # save the debug log of spark
    # extract the date
    result_df_date = result_df[['date']]
    prova = result_df_date['date'][0]
    for elem in prova:
        # print(type(elem)) # <class 'pyspark.sql.types.Row'>
        # print(type(elem['result'])) # <class 'str'>
        if "." in elem['result']:
            return date_to_utc(elem['result'], ".")
        if "/" in elem['result']:
            return date_to_utc(elem['result'], "/")
        
    return None

def search_date_pdf(dir_name, file_name):

    path_data = os.path.join(dir_name, file_name)
    
    # Creating a pdf file object
    pdfFileObj = open(path_data,'rb')
    
    # Creating a pdf reader object
    pdfReader = PyPDF2.PdfReader(pdfFileObj)
    
    # Getting number of pages in pdf file
    pages = len(pdfReader.pages)

    # Loop for reading all the Pages
    content = ""
    for i in range(pages):
        content += pdfReader.pages[i].extract_text() + "\n"
        content = " ".join(content.replace(u"\xa0", " ").strip().split())
    # print(content) # debug (full text of the pdf)

    # split the text by ' '
    list_content = content.split(" ")
    # print(list_content) # debug (text of the pdf in a list)
    print()

    # identify HEADER and FOOTER to exclude it by pdf text when searching for dates
    # 
    # header 2016-OJS066-114659-it.pdf -> ['GU/S', 'S3', '06/01/2016', '2872-2016-IT1', '/', '5'] -> 6 elements
    # remove page from position 3
    list_header = []
    j = 0
    for elem in list_content:
        if j == 6:
            break
        new_ele = elem
        if j == 3: # from position 3 remove the actual page number
            new_ele = new_ele[0:len(elem)-1]
        list_header.append(new_ele)
        j+=1
    # print(list_header) # debug

    # footer 2016-OJS066-114659-it.pdf -> footer: ['06/01/2016', 'S3', 'https://ted.europa.eu/TED1', '/', '5GU/S',] -> list_header[2] + list_header[1] + ted_url +  list_header[4] + list_header[5]
    list_footer = []
    list_footer.append(list_header[1])
    list_footer.append(list_header[2])
    list_footer.append(list_header[4])
    list_footer.append(list_header[5])
    list_footer.append(ted_url)
    # print(list_footer) # debug

    # Find the position of section "IV.3.8)"
    try:
        index = list_content.index("IV.3.8)")
        # print("index:", index) # debug
        text_part = ""
        for j in range(index+1, len(list_content), 1): # range(start, stop, step)
            text = list_content[j]
            list_text = []
            # print("text chunck:", text) # debug
            if "Luogo:" in text or "Sezione VI" in text: # it's in the next section: stop and search for the date (Sezione)
                # print(text_part) # debug
                list_text.append(text_part) # save the text to be analyzed in a list
                new_date = spark_date(list_text, file_name, annotator)
                # closing the pdf file object
                pdfFileObj.close()
                return new_date
            else:
                # concat the text to get the date
                if text not in list_footer and text not in list_header and "http" not in text and "GU/S" not in text and "-IT" not in text: # avoid text in header and footer
                    text_part +=  text + " "
                # print(text_part) # debug
    except ValueError:
        print("Section IV.3.8) not found")
        pdfFileObj.close()
        return None
    
    pdfFileObj.close()
    return None


### MAIN ###
print()
print()
print("*** Program start ***")
print()

t1 = datetime.now()
print("Starting time:", t1)
print()

# Prepare the output file

file_new_date_pdf = file_new_date_pdf.replace("#", str(annotator))
file_nlp_log = file_nlp_log.replace("#", str(annotator))

path_out_pdf = os.path.join(data_dir, file_new_date_pdf)
string_csv_header = "case_id;event;new_date;file_source"+os.linesep

with open(path_out_pdf, "w") as fp:
    fp.write(string_csv_header)

with open(file_nlp_log, "w") as fp:
    fp.write("Starting time:" + str(t1) + os.linesep)
    fp.write("******" + os.linesep)

# Open the complete log and get IT rows
path_data = os.path.join(data_dir, file_log)
dic_t = {'Case_ID':object}
df_log = pd.read_csv(path_data, sep = ";", dtype=dic_t, low_memory=False)
df_log_len = len(df_log)
# print(df_log.head())
df_log_country_cases = df_log['Case_ID'].nunique()
print("Log len:", df_log_len)
print("Distinct cases:", df_log_country_cases)
print()

# get distinct case_id
df_log_caseid = df_log[['Case_ID']].drop_duplicates()
# print("Log len:", len(df_log_caseid)) # --> df_log_country_cases

# Get the list of files in guue
path_guue_pdf = guue_dir + os.sep + "*.pdf"
list_pdf = glob.glob(path_guue_pdf)
# print(list_pdf) # debug

# Search data from case-id to 
i = 1
for row in df_log_caseid.itertuples():
    case_id = row[1]
    id = case_id[4:]
    padding = '0'
    id_0 = str(id.rjust(6, padding)) # the file .pdf has six digits with 0 padding prefix
    year = case_id[0:4]
    print("[", i, "/", df_log_len, "]")
    print("Case id:", case_id)
    print("Year:", year)
    print("ID:", id)
    print("ID-0:", id_0)
    # search in PDF
    for file in list_pdf:
        file_name = os.path.basename(file)
        # print(file_name) # debug
        parts_1 = file_name.split(".")
        parts_2 = parts_1[0].split("-")
        if parts_2[2] == id_0: # same id in file and case-id, search for dates
            print("Searching date in PDF file:", file_name)
            date_new = search_date_pdf(guue_dir, file_name)
            if date_new is None:
                print("No data foud:", file_name)
                break
            else:
                string_csv = str(case_id) + ";" + "BID-OPENING" + ";" + str(date_new) + ";" + file_name + os.linesep
                with open(path_out_pdf, "a") as fp:
                    fp.write(string_csv)
                break
    print()
    i+=1

print()
t2 = datetime.now()
print("Ending time:", )
delta = t2 - t1
print("Elapsed time:", delta)
print()

with open(file_nlp_log, "a") as fp:
    fp.write("******" + os.linesep)
    fp.write("Ending time:" + str(t2) + os.linesep)
    fp.write("Elapsed time:" + str(delta) + os.linesep)

print()
print("*** Program end ***")
print()