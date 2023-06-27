# News Sentiment Analysis and Stock market short sale data
# Short sale data source DE: https://www.bundesanzeiger.de/pub/de/nlp?6
# Short sale data source US: https://developer.finra.org/docs#query_api-equity-reg_sho_daily_short_sale_volume
import pandas as pd
import matplotlib.pyplot as plt
import requests 
import time

short_sale_data = pd.read_csv("//Users//Robert_Hennings//Dokumente//Uni//Master//2.Semester//MachineLearningWithTensorFlow//Project_Sentiment//ShortSaleData.csv")

short_sale_data.head(20)

# Next we need to group by individual stocks and create distinct time series for each unique stock ticker
# in order to be able to study them more closely later on with the specific news sentiment analysis
len(short_sale_data.Emittent.unique()) # 877 unique stocks listed 

short_sale_data.Emittent.unique()[0]



order = short_sale_data.Emittent.value_counts()[short_sale_data.Emittent.value_counts()>50]

short_sale_data[short_sale_data.Emittent == 'K+S Aktiengesellschaft']

# Replace the comma in the Position column by a .
# convert the date column to a datetime object
short_sale_data.Position = short_sale_data.Position.str.replace(",", ".").astype(float)
short_sale_data.Datum = pd.to_datetime(short_sale_data.Datum)

# Next step is to split the long data format into each distinct ticker and safe the single dataframes in a list
list_dfs = []
for ticker in short_sale_data.Emittent.unique():
    df = short_sale_data[short_sale_data.Emittent == ticker].reset_index(drop=True)
    list_dfs.append(df)


# Split data based on longest continouing time series
# Only consider series that have 30 following days




# Get the FINRA Data as well
import requests

startDate = "2020-01-01"
endDate = "2023-05-10"
limit = 5000
ticker = "GOOG"
groupName = "otcMarket"
datasetName = "regShoDaily"
# Datasets: https://developer.finra.org/catalog
# Field Description: https://www.finra.org/sites/default/files/2020-12/short-sale-volume-user-guide.pdf
# Single files: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files

# NCTRF Nasdaq TRF Carteret
# NQTRF Nasdaq TRF Chicago
# NYTRF NYSE TRF

def get_short_sale_dataFinra(start_date: str, end_date: str, limit: int, ticker: str, groupName: str, datasetName: str) -> pd.DataFrame:
    # Fix params
    startDate = start_date
    endDate = end_date

    url = f"https://api.finra.org/data/group/{groupName}/name/{datasetName}"

    headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
            }

    # more filters: https://developer.finra.org/docs#query_api-resource_endpoints-post_data
    customFilter = {
        "limits": limit,
        "async": "false",
        "compareFilters": [{"compareType": "equal",
                            "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                            "fieldValue": ticker}],
        "dateRangeFilters": [{
            "startDate": startDate,
            "endDate": endDate,
            "fieldName": "tradeReportDate"
            }]
    }


    request = requests.post(url, headers=headers, json=customFilter)

    data = pd.DataFrame.from_dict(request.json())

    return data


get_short_sale_dataFinra(startDate, endDate, limit, ticker, groupName, datasetName)


# Next group by unique date to sum up all the volume from each reporting facility
goog_short = get_short_sale_dataFinra(startDate, endDate, limit, ticker, groupName, datasetName)

goog_short.tradeReportDate = pd.to_datetime(goog_short.tradeReportDate)

aggFunc  = {"totalParQuantity": "sum",
            "shortParQuantity": "sum",
            "shortExemptParQuantity": "sum"}

aggData = goog_short.groupby(["tradeReportDate"]).agg(aggFunc)

# Next add the yahoo finance traded volume to the data
import yfinance as yf


goog = yf.Ticker("GOOG")
goog.history(start=startDate, end=endDate).Volume



# Also consider scraping the raw text files from the FINRA website
# Single files: https://www.finra.org/finra-data/browse-catalog/short-sale-volume-data/daily-short-sale-volume-files


def get_single_txt_filesFinra(startDate: str, endDate: str, datasetName: str) -> pd.DataFrame:
    # create daterange
    date_range = pd.date_range(start=startDate, end=endDate, freq="B")
    
    df = pd.DataFrame()
    counter = 0
    # build the url
    for d in date_range:
        counter +=1
        try:
            print(f"Loading data for day: {d.strftime('%Y%m%d')}, progress: {(counter/len(date_range)) * 100}%")
            url = f'https://cdn.finra.org/equity/regsho/daily/{datasetName}{d.strftime("%Y%m%d")}.txt'
            req = requests.request("GET", url)
            # time.sleep(2)
            txt = req.text.replace("\r\n", "|")
            for line in range(0,len(txt.split("|"))-5, 5):
                l = txt.split("|")[line:line+5]
                l_dict = {l[0]: l[1:]}
                df = df.append(pd.DataFrame.from_dict(l_dict, orient='index'))
            print(f"Finished loading and parsing data for day: {d.strftime('%Y%m%d')}")
        except:
            print(f"No txt file available for day: {d}")
        
    print("Finished Data Loading")
    df.columns = df.iloc[0, :]
    df.drop(["Date"], inplace=True)
    # Manipualate the index and transform it
    df.index = df.index.str[:4] + "-"+df.index.str[4:6] + "-"+df.index.str[6:]
    df.index = pd.to_datetime(df.index)

    return df

# Test the function
startDate = "01-04-2010"
endDate = "01-10-2010"
datasetName = "FNSQshvol"
# Possibel other ones: FNRAshvol, FNYXshvol, FORFshvol

test = get_single_txt_filesFinra(startDate, endDate, datasetName)

# Next we want to split the long version into a single timeseries for each individual ticker

def get_single_ticker_ts(master_df: pd.DataFrame, ticker_column_name: str, ticker: str) -> pd.DataFrame:
    tik_df = master_df[master_df[ticker_column_name] == ticker]

    return tik_df


master_df = test.copy()
ticker_column_name = "Symbol"
ticker = "AAPL"

get_single_ticker_ts(master_df, ticker_column_name, ticker)

# Loop over every unique ticker and create separate timeseries





# Next get (at least current) index constituents list, historical is hard to get
import finnhub
finnhub_client = finnhub.Client(api_key="chf4rd9r01qsph3d8ocgchf4rd9r01qsph3d8od0")
# Dax
print(finnhub_client.indices_const(symbol = "^GDAXI"))
# S&P
print(finnhub_client.indices_const(symbol = "^GSPCI"))

# Next get the specific data for only the constituents and perform analysis on them



short_sale_data.Positionsinhaber.unique()



# Get news data articles that cover the specific companies

from GoogleNews import GoogleNews
googlenews = GoogleNews(lang="en", start="01/01/2023", end='05/29/2023')


googlenews.get_news('APPLE')
googlenews.search('APPLE')

googlenews.total_count()

googlenews.results()[1]["link"]

googlenews.get_texts()


url = 'https://data.alpaca.markets/v1beta1/news'
params = {
    'start': '2021-12-28T00:00:00Z',
    'end': '2021-12-31T11:59:59Z',
    'symbols': 'AAPL,TSLA'
}
headers = {
    'Apca-Api-Key-Id': 'PKSL9Y1SCOF5381GMFJD',
    'Apca-Api-Secret-Key': '1FdtwPQOVrs5mFIORBt67HPFdfhpyxaTTIldZXbW'
}

response = requests.get(url, params=params, headers=headers)
print(response.json())