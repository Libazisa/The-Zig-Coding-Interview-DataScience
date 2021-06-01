# The-Data-Scientist-Coding-Interview-The-Zig
This contains the answer to the Data Science problems given by The Zig Consultancy.
# Task 1: Load in the dataset from the accompanying file "account-defaults.csv"
To perform this action, I chose to do the following

```Python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataframe = pd.read_csv('https://raw.githubusercontent.com/Libazisa/The-Zig-Coding-Interview-DataScience/master/account-defaults.csv')
```
When filling the dataframe in Python, we can either fill the N/A entries with 0s or drop those features entirely. Later on, we will see how this affects the performance of the algorithms the data will be subjected to. 

To fill the N/A entries with 0s we could chose to do the following.
```Python 
dataframe.fillna(0, inplace=True)
```

To fill the N/A entries entirely, we do the following;

```Python
dataframe.dropna(inplace=True)
```

# Task 2: Perform an exploratory data analysis on the accounts data
## In your analysis include summary statistics and visualizations of the distributions and relationships.
To see if there is any correlation between features, I chose to use the pearson correlation test. I then used the results of the Pearson correlation test to display a heatmap that would give us a better visual picture of correlations. 

```Python
import seaborn as sns
sns.heatmap(dataframe.corr(),cmap="Blues")
```
This then produces the following output

![PCT](https://user-images.githubusercontent.com/34988914/120101211-40437900-c145-11eb-8ea8-d5f12efe031e.png)

From the image displayed above, we see that the darker the hue, the more correlated the two variables (Dark blue having a pearson correlation score of 1) where as the lighter the hue, the less the correlation between the two variables (White having a pearson correlation score of 0). The image indicates that there a moderate correlation between the predictor variable (FirstYearDeliquency) and some other variables such as TotalInquirys, WorstDeliquency and HasInquiryTelecom. However in selecting a model, we will use all the variables for inout in predicting. 

# Task 3: Build one or more predictive model(s) on the accounts data using regression techniques
I will present two alternatives for the prediction task. Logistic regression and the random forest classifier. First we seperate the dataset into the independent and dependent variables as well as segmenting a test set from the dataset given. 
```Python
x = dataframe.iloc[:,1:9].values
y = dataframe.iloc[:,0].values
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20)
```
Next we preprocess the data to assist quick convergence of the logistic regression and random forest algorithms.

Next, we can build the logistic regression model.
```Python
from sklearn.linear_model import LogisticRegression
model_1 = LogisticRegression(solver='saga', max_iter=1000, random_state=42)
model_1.fit(x_train,y_train)
```
We can also implement the random forest algorithm on the same data by doing the following
```Python
from sklearn.ensemble import RandomForestClassifier
model_2 = RandomForestClassifier()
model_2.fit(x_train,y_train)
```
## Identify the strongest predictor variables and provide interpretations.
As can already be seen from the pearson correlation heatmap displayed above, the best predictors are TotalInquirys, WorstDeliquency and HasInquiryTelecom. 
## Identify and explain issues with the model(s) such as collinearity, etc.
## Calculate predictions and show model performance on out-of-sample data.
We can perform tests on both models using the test set. this can be seen by the following
```Python
from sklearn.metrics import classification_report
y_pred_1 = model_1.predict(x_test)
print('Performance of The Logistic Regression Model')
print(classification_report(y_test, y_pred_1))
y_pred_2 = model_2.predict(x_test)
print('Performance of the Random Forest Model')
print(classification_report(y_test,y_pred_2))
```
![Tests](https://user-images.githubusercontent.com/34988914/120312573-2fc40780-c2d9-11eb-9c54-252576fa768b.png)
We can also test the model with the features dropped instead of zeroed. 
From here we can see effect that zeroing or dropping features has on the model.
The macro average is the usual average we’re used to seeing. Just add them all up and divide by how many there were. Weighted average considers how many of each class there were in its calculation, so fewer of one class means that it’s precision/recall/F1 score has less of an impact on the weighted average for each of those things.
## Summarize out-of-sample data in tiers from highest-risk to lowest-risk.


