# Exploratory-Data-Analysis-on-Loan-defaulters-data

Summary of findings and proposals for futher data preprocessing and analyses

Understanding the DataSet :-

Our team worked on the dataset to know more about the data and to filter the dataset as
per our needs. So, we used a few basic commands to remove unwanted columns based on
null values. And used the describe method to know more statistical information about the
data. First, we used application_data.csv as ‘data(alias name) to work on.The columns
with 50% of null values were dropped.

There are two data sets
1) application_data.csv alias data
2) prervious_application.csv alias p_d

Cleaning the Data :-
1. The data is cleaned and columns with data anomalies are handled
2. The percentage of null Values in the data set for a particular column was above
50%. So those columns were Dropped
3. Then data below 13% null values were handled by replacing some the null values
with median in columns.
4. Rows are dropped which can create bias.
5. There are some irreverent columns in the dataset. They included flags for
documents, details etc. these columns were dropped from the dataset
6. Also converted integer columns into Categorical datatype. Object data type means
we have categorical data with 3 or less than 3 values
7. Also, for further analysis datatype of no columns is converted into numeric type.

EDA :-
Exploratory Data Analysis was done to check univariate analysis, categorical variables
and bivariate analysis. We can find that some numerical variables consisted of very high
values as compared to their respective means.
That's why we have created charts using boxplot to understand the patterns.From the boxplot
we were able to determine the outliers and now we are going to treat them.
Visualizations :-
For visualizations we have made Boxplots, pie charts, heatmaps, histograms and
line plots.

Observations from data analysis :
Dataset: application_data.csv

Univariate Analysis on the categorical variables:

1. clients with medium income are most likely to be default
2. clients with high credit are less likely to default
3. People who started their application on Sunday are less likely to default.
4. On the weekends clients tend to not fill applications. so the banks are less busy on
weekends.
5. People who are married tend to take more loans and also repay their loans as non
default. but maximum number of defaulters are present in married category
6. People who buy a house and come under secondary education tend to take out a loan.
7. females tend to take more loans compared to men.
8. people tend to take more loans in the form of cash.
9. resolving loans are less.

Univariate analysis on the continuous numerical variables:

1. People with lower total income are tending to default more
2. Lower the annuity higher the number of loans
3. New joiners have a higher tendency to take a loan
4. Application process is from 9 to 5, and the peak applications are during the hours 10 AM
to 12 AM
5. Normal families with children tend to take more loans.
6. People between the age of 25 to 45 tend to take more loans.
7. For less goods, people take more loans.

Bivariate Analysis on the continuous numerical variables:

1. Number of cash loans are more than the number of revolving loans.
2. Females tend to take more loans and repay on time. they also slightly tend to default
more than the males.
3. Unaccompanied people take more loans than other categories.
4. Working class people take more loans than other categories.
5. People who own a house/apartment have more tendency to take a loan and repay back.
6. Registration details after 4000 days of loan approval.

Merged dataset: (application_data.csv left-join previous_data.csv)

Insights from the univariate analysis of categorical variables in the merged data:

1. Older clients tend to take more loans and their loans are approved more also.
2. People who had taken loans with medium interest rate also tend to get their loan
approved in the current application.
3. The credit amount of the loan does not impact its approval
4. For clients with medium level income, the loan approval rate is the highest
5. In previous applications, the loans applied for on a Saturday had the highest approval
rate.
6. For current applications, loans filed on a Tuesday has the highest approval rate.
7. For both previous and current applications, loan applicants who were
unaccompanied/have taken loan singly, have had the highest approval rate.
8. Currently the bank is giving cash and revolving loans, but previously used to also give
consumer loans along with the other two types.
9. Previously consumer loans were the highest applied for, and currently it is cash loans.

Insights from the univariate analysis of continuous numerical variables in the merged data:

1. A large number of applications are filed between 9 to 2 pm for both current and
previous data.
2. Hence, we can say that the banks have their busiest hours from 9 to 2 pm.
3. Applicant(s) with family tend to take more loans.
4. For consumer loans, previously banks had high unused loan offers but currently the
number of refused loans are high.
5. Previously banks had high unused loan offers and currently canceled/refused offers are
similar for loan annuity.
6. Previously banks had high unused offers of the credit amount given and currently a
high number of refusals for credit amount related requests.

Insights from the bivariate analysis of categorical variables in the merged data:

1. Cash loans have the highest count of Approved loans.
2. Highest number of approvals for working applicants.
3. Highest number of approvals for Secondary/secondary special educated applicants.
4. Highest number of approvals for Married applicants.
5. Highest number of approvals for House/apartment owners.
6. Highest number of approvals for Consumer Loans.
7. repeated applications got approved the most number of times.

Insights from the bivariate analysis of continuous numerical variables in the merged data:

1. Credit amounts requested in previous applications have highest refusals and credit
amounts for the current application do not show this trend.
2. Time spent in unused offers is higher as compared to other categories.
3. So banks should reduce time spent on unused offers.
4. Nuclear families(2-3 people in a family) get the highest approval.
5. Previously most of the applications were canceled or refused.
6. But now Refused/Canceled/Approved/Unused all four have a similar situation for
credits amounts requested for consumer loans.
7. Previously most of the applications were canceled or refused
8. But now Refused/Canceled/Approved/Unused all four have a similar situation for loan
annuities.

Suggestions for the Bank / Conclusion :-

Target/focused variable for Application dataset - TARGET
Target/focused variable for Previous dataset - NAME_CONTRACT_STATUS
Top Major variables to consider for loan prediction:
NAME_EDUCATION_TYPE
AMT_INCOME_TOTAL
DAYS_BIRTH
AMT_CREDIT
DAYS_EMPLOYED
AMT_ANNUITY
NAME_INCOME_TYPE
CODE_GENDER
NAME_HOUSING_TYPE

The above mentioned variables are to be considered before approving applications due to the risk
of loss!

In the future, if we want to decrease the defaulted cases and increase the chances of repayment
then we need to target the people with these following traits and be cautious while dealing with
them:

1. Freshers who got placed recently.
2. Families with children and with work experience of more than a decade and living in their
own home.
3. People with graduation from top colleges.
4. People with no prior loan records.
5. Working class people.

Proposal of method to handle the problem of class imbalance in the dataset:

During the exploration of the data, we found that the dataset is highly imbalanced. ~91% of the
data corresponds to people who did not default on the loan,and ~9% of the data corresponds to
people who defaulted on the loan. So, using such a dataset to train machine learning models for
predicting the default cases in the real world would not be a good idea as the trained models
would be unreliable and biased towards the class of data with higher occurrence(here, it is the
data pertaining to non-defaulters). A way to reduce the high class imbalance ratio in the dataset
would be to use a GAN model(Generative Adversarial Network) to generate synthetic records
for the class of data with less occurrence (here, it is the records of people who defaulted on the
loan). Then, we augment the synthetic data to the original dataset. A decision should be made
beforehand on the number of synthetic records that needs to be generated, which should be high
enough to substantially reduce the imbalance ratio, but not too high that it results in overfitting
the machine learning models trained on it.
Proposal of machine learning model to perform predictive analytics:
Next, we can train a Random forest classifier on the augmented dataset for the purpose of
predictive analytics, primarily to predict people who may default in the future based on their past
history. This is because, we found that the correlation factors between the different
dimensions(variables) of the dataset are not similar, hence applying linear/logistic regression in
this case would be inefficient. A random forest classifier would suffice here because it gives the
best average output from multiple decision tree outputs, where the individual decision tree
outputs are specific and low on bias.
