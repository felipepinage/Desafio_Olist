import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('olist_prepared.csv', sep=',', header=0).replace(np.NaN, 0)  # read csv file
file = open("olist_delay_results.txt","w");

df = df[df['delta_purch_delivered'] > 0] #remove instances with delivery before purchase (outliers)
df['delay'] = df['delta_purch_delivered'] - df['delta_est_delivered'] #calculate delivery delay and include as a column

df_ind = df.iloc[:,1:12] #dataframe only with independent variables
df_ind['most_voted_class'] = df['most_voted_class'] #include class column

df_review = df[['delta_purch_delivered',
                'delta_est_delivered',
                'delta_review_date',
                'delta_answer_date',
                'most_voted_class',
                'delay']] #dataframe only with dependent variables

delay_ind = df_ind[df["delay"] > 0 ] #get only customer with delivery delayed and independent variables
corr = delay_ind.corr() #calculate correlation between independent variables
sns.heatmap(corr, annot=True, fmt='.3f') #plot heatmap
corr = abs(corr)
feat_imp = corr['most_voted_class'].nlargest(4) #3 more important features when delivery delayed
keys_imp = feat_imp[1:].keys()
file.write('THREE MOST IMPORTANT REASONS TO DEFINE CUSTOMER SATISFACTION WHEN DELIVERY IS DELAYED\n'
            '%s - %.3f,\n'
            '%s - %.3f,\n'
            '%s - %.3f.\n' % (keys_imp[0], feat_imp[1], keys_imp[1], feat_imp[2], keys_imp[2], feat_imp[3]))

total_delays = delay_ind.groupby("most_voted_class").count()['order_status'] #number of delays per class
total_delays_perc = (total_delays*100)/total_delays.sum() #percentage of delays per class
file.write("\nTOTAL DELAYS - %d\n"
            "Delivery problem: %d (%.2f%%),\n"
            "Quality problem: %d (%.2f%%),\n"
            "Satisfied: %d (%.2f%%).\n"
            % (total_delays.sum(), total_delays[0], total_delays_perc[0], total_delays[1], total_delays_perc[1], total_delays[2], total_delays_perc[2]))


delay_dep = df_review[df_review["delay"] > 0 ] #dataframe contains only instances with delayed delivery and dependent variable
rbd_delay = delay_dep[delay_dep["delta_review_date"] < 0 ].count()['delta_review_date'] #number of reviews before delivery (delayed)
rad_delay = delay_dep['delay'].count() - rbd_delay #number of reviews after delivery (delayed)
perc_rbd_delay = (rbd_delay*100)/delay_dep['delay'].count() #percentage of review before delivery (delayed)
perc_rad_delay = 100 - perc_rbd_delay #percentage of review after delivery (delayed)
file.write('\nDELAYED DELIVERY - NUMBER OF REVIEWS BEFORE AND AFTER DELIVERY\n'
            '%d (%0.2f%%) of customer make a review BEFORE delivery,\n'
            '%d (%0.2f%%) of customer make a review AFTER delivery.\n'
            % (rbd_delay, perc_rbd_delay, rad_delay, perc_rad_delay))

rbd_delay_class = delay_dep[delay_dep["delta_review_date"] < 0 ].groupby('most_voted_class').count()['delta_review_date'] #number of reviews before delivery per class
perc_rbd_delay_class = (rbd_delay_class*100)/rbd_delay #percentage of reviews before delivery per class
file.write("\nDELAYED DELIVERY - NUMBER OF REVIEWS BEFORE DELIVERY (PER CLASS) - %d\n"
            "Delivery problem: %d (%.2f%%),\n"
            "Quality problem: %d (%.2f%%),\n"
            "Satisfied: %d (%.2f%%).\n"
            % (rbd_delay, rbd_delay_class[0], perc_rbd_delay_class[0], rbd_delay_class[1], perc_rbd_delay_class[1], rbd_delay_class[2], perc_rbd_delay_class[2]))

rad_delay_class = delay_dep[delay_dep["delta_review_date"] >= 0 ].groupby('most_voted_class').count()['delta_review_date'] #number of reviews after delivery per class
perc_rad_delay_class = (rad_delay_class*100)/rad_delay #percentage of reviews after delivery per class
file.write("\nDELAYED DELIVERY - NUMBER OF REVIEWS AFTER DELIVERY (PER CLASS) - %d\n"
            "Delivery problem: %d (%.2f%%),\n"
            "Quality problem: %d (%.2f%%),\n"
            "Satisfied: %d (%.2f%%)."
            % (rad_delay, rad_delay_class[0], perc_rad_delay_class[0], rad_delay_class[1], perc_rad_delay_class[1], rad_delay_class[2], perc_rad_delay_class[2]))

file.close();