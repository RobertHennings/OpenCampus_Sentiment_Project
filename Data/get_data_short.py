import pandas as pd
import requests
import time
import glob

def get_single_txt_filesFinra(startDate: str, endDate: str, datasetName: str, save_local: bool, save_path: str, override_files: bool) -> pd.DataFrame:
    """Scrapes single .txt file from the Finra website and appends the single days to a msterdataframe
       that will be returned, optional saves the single files at the specified path

    Args:
        startDate (str): start date from which onwards to retrieve the data
        endDate (str): end date to which the single text files should be scraped
        datasetName (str): Dataset from which specific reporting entity to retrieve
        save_local (bool): Determines if a local .csv copy of the parsed file should be saved at the specified file path
        save_path (str): The specified file path to save the .csv files in
        override_files (bool): Determines if the already existing files should be overridden or not

    Returns:
        pd.DataFrame: Masterdataframe that holds all the single days
    """
    # create daterange
    date_range = pd.date_range(start=startDate, end=endDate, freq="B")

    df_master = pd.DataFrame()
    counter = 0
    # build the url
    for d in date_range:
        df = pd.DataFrame()
        counter +=1
        try:
            print(f"Loading data for day: {d.strftime('%Y%m%d')}, progress: {(counter/len(date_range)) * 100}%")
            url = f'https://cdn.finra.org/equity/regsho/daily/{datasetName}{d.strftime("%Y%m%d")}.txt'
            #url = f'https://cdn.finra.org/equity/regsho/daily/{datasetName}{"20110804"}.txt'
            req = requests.request("GET", url)

            txt = req.text.replace("\r\n", "|")
            if date_range[500] >= pd.Timestamp("2011-08-04"):
                leng = 6
            else:
                leng = 5

            for line in range(0,len(txt.split("|"))-5, leng):
                l = txt.split("|")[line:line+5]
                l_dict = {l[0]: l[1:]}
                df = df.append(pd.DataFrame.from_dict(l_dict, orient='index'))

            df.columns = df.iloc[0, :]
            df.drop(["Date"], inplace=True)
            df_master = df_master.append(df)

            if save_local:
                if override_files:
                    print(f"Saving file locally: {d.strftime('%Y%m%d')}.csv")
                    df.to_csv(save_path + "//" + d.strftime('%Y%m%d') + ".csv")
                else:
                    if d.strftime('%Y%m%d') + ".csv" not in glob.os.listdir(save_path):
                        df.to_csv(save_path + "//" + d.strftime('%Y%m%d') + ".csv")

            print(f"Finished loading and parsing data for day: {d.strftime('%Y%m%d')}")
        
        except:
            print(f"No txt file available for day or error at: {d.strftime('%Y%m%d')}")

    print("Finished Data Loading")
    # Manipualate the index and transform it
    df_master.index = df_master.index.str[:4] + "-"+df_master.index.str[4:6] + "-"+df_master.index.str[6:]
    df_master.index = pd.to_datetime(df_master.index)

    return df

# Still failures in files: 20100505, 20100506, 20100507, 20100514
def get_single_ticker_ts(master_df: pd.DataFrame, ticker_column_name: str, ticker: str) -> pd.DataFrame:
    tik_df = master_df[master_df[ticker_column_name] == ticker]

    return tik_df