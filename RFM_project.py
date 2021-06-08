###############################################################
# RFM ile Müşteri Segmentasyonu (Customer Segmentation with RFM)
###############################################################


###############################################################
# 1. Veriyi Anlama ve Hazırlama
###############################################################
import numpy as np
import datetime as dt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

# 2009-2010 yılı içerisindeki veriler
df_ = pd.read_excel("online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()
df.head()
df.shape
def check_df(dataframe, head=5):
    """
    Shows that dataframe's shape,types,first 5 variables, last 5 variables, missing values and dataframe's quantiles.
    Parameters
    ----------
    dataframe: dataframe
        dataframe from which variable names are taken
    head: int, optional
        the first 5 elements of the df

    Returns
    -------
    There are no variables to return

    Example
    -------
    import seaborn as sns
    df = sns.load_dataset("tips")
    check_df(df)

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df, 5)

###############################################################
# 3. Veri Hazırlama (Data Preparation)
###############################################################

df.shape
# Q3
df.isnull().any()
df.isnull().sum()
# Q4
df.dropna(inplace=True)
df.isnull().sum()
df.shape


# Q5 Eşsiz ürün sayısı kaçtır?
df["StockCode"].nunique()
# 4070
#Q6 Hangi üründen kaçar tane vardır?
df["StockCode"].value_counts().head()
Out[11]:
85123A    2313
22423     2203
85099B    2159
47566     1727
20725     1639
#Q7 En çok sipariş edilen 5 ürünü çoktan aza doğru sıralayınız.
df["StockCode"].value_counts().sort_values(ascending=False).head()
85123A    2313
22423     2203
85099B    2159
47566     1727
20725     1639
#Q8 Faturalardaki ‘C’ iptal edilen işlemleri göstermektedir. İptal edilen işlemleri veri setinden çıkartınız.
df = df[~df["Invoice"].str.contains("C", na=False)]
df.head()
#Q9 Fatura başına elde edilen toplam kazancı ifade eden ‘TotalPrice’ adında bir değişken oluşturunuz.
df["TotalPrice"] = df["Quantity"] * df["Price"]
df.head()
###############################################################
# 4. RFM Metriklerinin Hesaplanması (Calculating RFM Metrics)
###############################################################

# Recency(Yenilik) : Müşterinin son satın almasından bugüne kadar olan geçen süre
# Frequency(Sıklık) : Toplam satın alma sayısı
# Monetary( Parasal değer) : Müşterinin yaptığı toplam harcama

df["InvoiceDate"].max()

today_date = dt.datetime(2011, 12, 11)

# recency
# frequency
# monetary

rfm = df.groupby('Customer ID').agg({'InvoiceDate': lambda InvoiceDate: (today_date - InvoiceDate.max()).days,
                                     'Invoice': lambda Invoice: Invoice.nunique(),
                                     'TotalPrice': lambda TotalPrice: TotalPrice.sum()})

rfm.head()
rfm.columns = ['recency', 'frequency', 'monetary']
rfm.describe().T
rfm = rfm[rfm["monetary"] > 0]
rfm.head()



###############################################################
# TASK 3
###############################################################

# Recency
rfm["recency_score"] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
# 0,20,40,60,80,100

rfm["frequency_score"] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

rfm["monetary_score"] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])


rfm["RFM_SCORE"] = (rfm['recency_score'].astype(str) +
                    rfm['frequency_score'].astype(str))

rfm.head()

rfm['recency'].head()

###############################################################
# TASK 4
###############################################################


seg_map = {
    r'[1-2][1-2]': 'hibernating',
    r'[1-2][3-4]': 'at_Risk',
    r'[1-2]5': 'cant_loose',
    r'3[1-2]': 'about_to_sleep',
    r'33': 'need_attention',
    r'[3-4][4-5]': 'loyal_customers',
    r'41': 'promising',
    r'51': 'new_customers',
    r'[4-5][2-3]': 'potential_loyalists',
    r'5[4-5]': 'champions'
}


rfm['segment'] = rfm['RFM_SCORE'].replace(seg_map, regex=True)

rfm[["segment", "recency", "frequency", "monetary"]].groupby("segment").agg(["mean", "count"])

rfm.head()



# TASK 5
# champions, can't loose, need attention

champions = rfm[rfm['segment'] == 'champions']
cant_loose = rfm[rfm['segment'] == 'cant_loose']
need_attention = rfm[rfm['segment'] == 'need_attention']

# cant loose
# 63 müşteri, total monetary 2796. 132 günde bir 8 kez alışveriş yapmışlar.
# Bu müşterilere özel kampanyalar yapılabilir. Hesaplarına kuponlar tanıtılabilir. Sizi özledik diye bir pop up çıkabilir.
cant_loose[['recency','frequency','monetary']].agg(['mean', 'count'])

#champions
# 633 müşteri, yaklaşık 6 günde bir 12 kez alışveriş yaparak 6857.96 birim kazandırmışlar.
# Bu müşterilere özel aramalar yapılabilir. Hediye çeki. 1 alana 1 bedava tarzında kampanyalar yapılabilir.
champions[['recency','frequency','monetary']].agg(['mean', 'count'])

# need_attention
# 187 müşteri, 52 günde bir 2 kez alışveriş yaparak 897.63 birim kazandırmışlar.
# Sizi özledik tarzında pop-uplar çıkabilir. Sürekli hatırlatıcı bildirimler gönderilebilir. Daha önceki alışverişine özel kampanyalar yapılabilir.
need_attention[['recency','frequency','monetary']].agg(['mean', 'count'])

loyal_df = pd.DataFrame()
loyal_df["loyal_customer_id"] = rfm[rfm["segment"] == "loyal_customers"].index
loyal_df.head()

loyal_df.to_excel("loyal_customers.xlsx", sheet_name='Loyal Customers Index')