from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np

df= pd.read_csv(r'grocc.csv')  #import the dataset

dataset = df.dropna(thresh=2,axis=0)  #drop rows that has only one true value
dataset = dataset.replace(' ', np.nan, regex=True)  #filling blank spaces with nan
dataset = dataset.to_numpy()   #convert the dataframe into a numpy array


#Creating the dataframe of frequent itemsets
te1 = TransactionEncoder()
te_ary = te1.fit(dataset.astype(str)).transform(dataset.astype(str))
df = pd.DataFrame(te_ary, columns=te1.columns_)


#Apply Apriori algorithm and identify the itemsets where the support is greater than 10%.
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)


#Find the set of Association rules using the metric ’lift’
rules= association_rules(frequent_itemsets, metric="lift", min_threshold=1)

print(rules)


#part(e) answer --   Let us take the first rule.
#An antecedent is an item found within the data. A consequent is an item found in combination with the antecedent.
#It is like an IF and THEN relationship. We can see in the first one the antecedent is 'pastry' and consequent is null. That means there are no consequents.
#'antecedent support' computes the proportion of transactions that contain the antecedent. This is an indication of how frequently the item appears in the data.
#Here that value is 0.109186. We refer to an itemset as a "frequent itemset" if support is larger than a specified minimum-support threshold.
#'Lift' is used to measure how much more often the antecedent and consequent of a rule occur together than we would
#expect if they were statistically independent. If antecedent and consequent are independent, the Lift score will be exactly 1. Here it is 1.0 because there are no
#consequents.
#'Leverage' computes the difference between the observed frequency of antecedent and consequent appearing together and the frequency that would be expected if they
#were independent. Leverage value of 0 indicates independence. Here it is 0 because there are no consequents.
#'Conviction' is defined as; conviction(A→C)=1−support(C)/1−confidence(A→C), A high conviction value means that the consequent is highly depending on the antecedent.
#If confidence score is 1, the denominator becomes 0. Then the conviction score is defined as 'inf'. If items are independent, the conviction is 1.
#Here convication is 'inf'.



#part(f) answer -- 
#we can get the rules by changing the association_rules() funtion as below;
#rules= association_rules(frequent_itemsets, metric="lift", min_threshold=4)  This gives an empty dataset.
#rules= association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8) This gives 5 association rules
#if we combine this together it gives an empty dataframe. zero rules



