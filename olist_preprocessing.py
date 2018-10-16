import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# PRE-PROCESSAMENTO: codificar os dados categóricos
def codificaCat(datafm):
    cols = datafm.columns
    num_cols = datafm._get_numeric_data().columns
    cat = list(set(cols) - set(num_cols))
    for i in range(cat.__len__()):
        labelEncoderRating = LabelEncoder()
        datafm[cat[i]] = labelEncoderRating.fit_transform(datafm[cat[i]])

# ELIMINA DUPLICATAS; CALCULA NOVAS CARACTERÍSTICAS E PREPARA UM DATASET PARA CLASSIFICAÇÃO
def prepareDataset(df):
    df.groupby("order_aproved_at").size().reset_index(name='count').sort_values("count", ascending=False)

    if 'most_voted_class' in df.columns:
        df = df[df['most_voted_class'] != 0] #remove instances with NaN label
    df.drop_duplicates(subset="order_aproved_at", inplace=True)
    df.count()

    df_2 = df[["order_purchase_timestamp",
               "order_estimated_delivery_date",
               "order_delivered_customer_date",
               "review_creation_date",
               "review_answer_timestamp",
               "customer_state"]]

    df_2.dropna(inplace=True)  # remove missing values

    df_2['date_purchase'] = pd.to_datetime(df_2["order_purchase_timestamp"])
    df_2['date_approved'] = pd.to_datetime(df['order_aproved_at'])
    df_2['date_estimated'] = pd.to_datetime(df_2['order_estimated_delivery_date'])
    df_2['date_delivered'] = pd.to_datetime(df_2['order_delivered_customer_date'])
    df_2['date_review'] = pd.to_datetime(df_2['review_creation_date'])
    df_2['date_answer'] = pd.to_datetime(df_2['review_answer_timestamp'])

    # days to be delivered
    df_2['delta_purch_delivered'] = (df_2['date_delivered'] - df_2['date_purchase']).dt.days
    # estimated delivery days
    df_2['delta_est_delivered'] = (df_2['date_estimated'] - df_2['date_purchase']).dt.days
    # days to approve the purchase
    df_2['delta_approve_purch'] = (df_2['date_approved'] - df_2['date_purchase']).dt.days
    # days to review creation
    df_2['delta_review_date'] = (df_2['date_review'] - df_2['date_delivered']).dt.days
    # days to answer research
    df_2['delta_answer_date'] = (df_2['date_answer'] - df_2['date_delivered']).dt.days

    df_3 = df[['order_status',
               'order_products_value',
               'order_freight_value',
               'order_items_qty',
               'order_sellers_qty',
               'customer_city',
               'customer_state',
               'product_category_name',
               'product_name_lenght',
               'product_description_lenght',
               'product_photos_qty',
               'review_score']]

    if 'most_voted_class' in df.columns:
        df_3 = df_3.join(df['most_voted_class'], lsuffix='_dat', rsuffix='_lab') #includes 'most_voted_class column to df_3

    df_4 = df_3.join(df_2[['delta_purch_delivered',
                           'delta_est_delivered',
                           'delta_approve_purch',
                           'delta_review_date',
                           'delta_answer_date']], lsuffix='_origin', rsuffix='_calc') #join df_2 and df_3

    df_4 = df_4[df_4['delta_purch_delivered'] > 0] #remove instances whose delivery date is before purchase date

    return (df_4)

df = pd.read_csv('olist_classified_public_dataset.csv', sep=',', header=0).replace(np.NaN, 0)  # read csv file

df_prepared = prepareDataset(df) #Prepare dataset to build a model

codificaCat(df_prepared);  #Codify categorical data

df_prepared.to_csv("olist_prepared.csv") #Save processed data to a new dataset
