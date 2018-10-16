# Desafio_Olist

STEP 1: olist_preprocessing.py

- From the original classified dataset, the following attributes remain in the output dataset: 
'order_status', 'order_products_value', 'order_freight_value', 'order_items_qty', 'order_sellers_qty', 'customer_city', 'customer_state', 'product_category_name', 'product_name_lenght', 'product_description_lenght', 'product_photos_qty', 'review_score'.

- and the following attributes are only used to provide NEW ATTRIBUTES:
'order_purchase_timestamp', 'order_aproved_at', 'order_estimated_delivery_date', 'order_delivered_customer_date', 'review_creation_date', 'review_answer_timestamp'

- NEW ATTRIBUTES (included in the output dataset):
'delta_purch_delivered': days to be delivered from the purchase date
'delta_est_delivered': estimated delivery days from the purchase date
'delta_approve_purch': days to approve the purchase
'delta_review_date': days to review creation from the delivery date
'delta_answer_date': days to research answer from the delivery date

- input: [olist_classified_public_dataset.csv]
- output: [olist_prepared.csv]

--------------------------------------------------------------------------------------------------------------------------
STEP 2: olist_stream.py

- This script analyze the dataset to identify if the data distribution is stationary or non-stationary.
- The prequential accuracy is evaluated to find DRIFTS (change points)
- The points of DRIFT DETECTION are noticed and quantified in 'stream_results.txt'.
- In case of non-stationary data distribution, it means that a single prediction model may not present a good performance to the whole dataset.

- input: [olist_prepared.csv]
- output: [olist_stream_results.txt], [Stream_Accuracy.png]

--------------------------------------------------------------------------------------------------------------------------
STEP 3: olist_modelo.py

- This script buils a decision model to predict if the customer answer {delivery_problem, quality_problem, satisfied}
- Oversampling technique is used to balance the classes.
- Ensemble Classifiers used: Random Forest
- Sampling: 60% to train, 40% to test
- Selection of most important features is used to refine the decision model.
- Evaluation metrics: accuracy, precision, recall, f1score, and confusion matrix
- The decision model is saved as Olist_Model.pkl (inside Olist_Model.rar)

- input: [olist_prepared.csv]
- output: [olist_modelo_results.txt], [Conf_Matrix_Limited_Model.png], [Conf_Matrix_Original_Model.png], [Olist_Model.pkl]

--------------------------------------------------------------------------------------------------------------------------
STEP 4: olist_delay_analisys.py

- This script is a basic analysis of how the customer review the order in case of delayed deliveries.
- It presents:
(THREE MOST IMPORTANT REASONS TO DEFINE CUSTOMER SATISFACTION WHEN DELIVERY IS DELAYED), 
(TOTAL DELAYS), 
(DELAYED DELIVERY - NUMBER OF REVIEWS BEFORE AND AFTER DELIVERY), 
(DELAYED DELIVERY - NUMBER OF REVIEWS BEFORE DELIVERY (PER CLASS)), 
(DELAYED DELIVERY - NUMBER OF REVIEWS AFTER DELIVERY (PER CLASS)).

- Conclusion: 
When delivery is delayed, customers tend to immediately (before delivery) review the order as "delivery problem", only 12% of these customers wait for delivery to make a fair review. In addition, even delayed, some customers review the order as "satisfied", it may be related to their satisfaction with three other features (products value, freight value, and products description lenght). Finally, reviews before delivery assigned as "quality problem" are not considered fair reviews, and it may due to the customer insatisfactions with the same three other features (products value, freight value, and products description lenght).

- input: [olist_prepared.csv]
- output: [olist_delay_results.txt]
