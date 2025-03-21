import pandas
import pandas as pd
from apyori import apriori

def categorize_month(dt: pd.Timestamp):
    return dt.month

def categorize_time(dt: pd.Timestamp):
    hour = dt.hour
    day_type = ""
    # day_type = "week_day"
    # if dt.day_of_week == 5 or dt.day_of_week == 6:
    #     day_type = "weekend_day"
    if 0 <= hour < 6:
        return f'time:00:00-5:59 [{day_type}]'
    elif 6 <= hour < 12:
        return f'time:06:00-11:59 [{day_type}]'
    elif 12 <= hour < 16:
        return f'time:12:00-15:59 [{day_type}]'
    elif 16 <= hour < 19:
        return f'time:16:00-18:59 [{day_type}]'
    elif 19 <= hour <= 24:
        return f'time:19:00-24:00 [{day_type}]'
    else:
        return hour

class Dataset:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path, parse_dates=["InvoiceDate"])
        self.stock_code_to_description = self.df.groupby("StockCode").agg({"Description": "first"}).to_dict()["Description"]
        self.csv_path = csv_path

    def get_datetime_tuples(self):
        invoice_datecat_tuples = list(zip(self.df['InvoiceDateCat'], self.df['StockCode']))
        return invoice_datecat_tuples

    def get_grouped_data(self, group, data_item):
        orders_depending_on_invoice = self.df.groupby(group)[data_item].apply(lambda x: list(set(x))).tolist()
        return orders_depending_on_invoice

    def handle_missing_or_wrong_values(self):
        #remove empty rows
        self.df = self.df[self.df["Invoice"] != ""]
        self.df = self.df[ self.df["StockCode"] != ""]
        self.df = self.df[self.df["Description"] != ""]
        self.df = self.df[self.df["Quantity"] != ""]
        self.df = self.df[self.df["InvoiceDate"] != ""]
        self.df = self.df[self.df["Price"] != ""]
        self.df = self.df[self.df["Customer ID"] != ""]
        self.df = self.df[self.df["Country"] != ""]

        self.df = self.df[self.df["InvoiceDate"] < pd.Timestamp('today')]#remove purchases in the future
        self.df = self.df[self.df.Invoice.str.isnumeric()]#remove all Invoices that are not numbers
        #self.df = self.df[self.df.StockCode.str.isnumeric()]#remove all StockCodes that are not numbers not all stock codes are intigers
        self.df = self.df[pd.to_numeric(self.df.Price, errors='coerce').notnull()]
        self.df["Customer ID"] = pd.to_numeric(self.df["Customer ID"], errors="raise", downcast='integer')
        self.df = self.df[pd.to_numeric(self.df["Customer ID"], errors='coerce', downcast='integer').notnull()]
        self.df = self.df[self.df.duplicated(subset=["StockCode", "Description"], keep=False)]#remove inconsistent row between stockcode and description

    def categorize_data(self):
        date_categories = ["morning", "noon", "afternoon", "evening", "night"]
        unit_price_categories = ["cheap", "normal", "expensive", "extremely_expensive"]
        quantity_categories = ["one", "multiple", "lot"]
        self.df['InvoiceDateCat'] = self.df['InvoiceDate'].apply(categorize_time)
        self.df['quantityCat'] = pd.qcut(self.df['Quantity'], 3, labels=["small", "normal", "many"])
        self.df['priceCat'] = pd.qcut(self.df['Price'], 4, labels=["cheap", "normal", "expensive", "extremely_expensive"])
        self.df['Month'] = self.df['InvoiceDate'].apply(categorize_month)

    def keep_last_x_items(self,x):
        self.df = self.df.sort_values('InvoiceDate').tail(x)