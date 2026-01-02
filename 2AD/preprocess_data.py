import pandas as pd

file1 = "EP6_RCVs_2022_06_13.xlsx"
file2 = "EP6_Voted docs.xlsx"
file3 = "EP7_RCVs_2014_06_19.xlsx"
file4 = "EP7_Voted docs.xlsx"
file5 = "EP8_RCVs_2019_06_25.xlsx"
file6 = "EP8_Voted docs.xlsx"
file7 = "EP9_RCVs_2022_06_22.xlsx"
file8 = "EP9_Voted docs.xlsx"

def read_excel_to_df(file, usecols=None, sheet=None):
    df = pd.read_excel(file, usecols=usecols)
    return df

def is_RCV_or_voted_docs(file):
    if "RCV" in file:
        return "RCV"
    elif "Voted docs" in file:
        return "Voted docs"
    
def read_pickled_files(file):
    df = pd.read_pickle(file + ".pkl")
    return df

def pickle_xsl_files(filesRCV):

    for file in filesRCV:
        df = read_excel_to_df("votewatch/" + file)
        df.to_pickle("pickled_data/" + file + ".pkl")



if __name__ == "__main__":
    file_names = [
        file1,
        file2,
        file3,
        file4,
        file5,
        file6,
        file7,
        file8
    ]

   # all files (if needed)
    all_files = {file: is_RCV_or_voted_docs(file) for file in file_names}

    # filter out RCV files
    filesRCV = []
    for file in all_files:
        if all_files[file] == "RCV":
            filesRCV.append(file)
    