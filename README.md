# Robust-Regression

## 1. Introduction
Regression models are used to predict a numerical value (dependent variable) given a set of input variables (independent variables). The most famous model of the family is the linear regression [2].
Linear regression fits a line (or hyperplane) that best describes the linear relationship between some inputs (X) and the target numeric value (y).
However, if the data contains outlier values, the line can become biased, resulting in worse predictive performance. Robust regression refers to a family of algorithms that are robust in the presence of outliers.


## 2. Linear Regression With Outliers
### 2.1 What is an outlier?
Linear regression models assume that each independent variable follows a Gaussian distribution. A factor that can affect the probability distribution of the variables when using a linear regression model is the presence of outliers. Outliers are observations (samples) that are far outside the expected distribution.
For example, if a variable follows the normal distribution , then an observation that is 3 (or more) standard deviations far from the mean is considered an outlier. So, a dataset having outliers can cause problems to a linear regression model.

### 2.2 How can outliers cause problems ?
Outliers in a dataset can bias summary statistics calculated for the variable (e.g. the mean and standard deviation). This results in models that are not performing well and that are highly biased and influenced by the underlying outliers.
To deal with the presence of outliers in our dataset, we can use a family of robust regression models. These models are known as robust regression algorithms. The two most famous robust regression algorithms are the Random Sample Consensus Regression(RANSAC) and the Huber Regression.

### 2.3 RANSAC Regression
Random Sample Consensus (RANSAC) is a well-known robust regression algorithm 
RANSAC tries to separate data into outliers and inliers and fits the model only on the inliers.
In this article we will only use RANSAC but almost all statements are true for the Huber Robust regression as well.

## 3. Working example
### 3.1.The artificial dataset
First, to illustrate the difference between the regular linear regression and the RANSAC robust model, we will create some data that have outliers. The example is based on the examples from the sklearn documentation page.
Our artificial dataset will consist of: one dependent variable (y) and one independent variable (X) with 1000 observations from which 50 are outliers.
```
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model, datasets
n_samples = 1000
n_outliers = 50
X, y, coef = datasets.make_regression(n_samples=n_samples, n_features=1, n_informative=1, noise=10, coef=True, random_state=0)
# Add outlier data
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)
print("The independent variable X has {} observations/samples".format(X.shape[0]))
print("The dependent variable y has shape {}".format(y.shape))
```

Let’s also plot the data to visualize the artificial data and see the outliers.
```
plt.scatter(X,y)
plt.show()
```


It is clear that we have 1000 observations from which 50 are outliers (right cloud of points).

### 3.2. The model fitting
Now let’s fit a regular regression model and a robust model on the data and check the estimated regression coefficients.

```
# Fit line using all data
lr = linear_model.LinearRegression()
lr.fit(X, y)
# Robustly fit linear model with RANSAC algorithm
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
# Predict data of estimated models
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)
# Compare estimated coefficients
print(“Estimated coefficients (true, linear regression, RANSAC):”)
print(coef, lr.coef_, ransac.estimator_.coef_)
```

We observe something very interesting here. The true coefficient is 82.19 while the estimated by the regural regression is 54.17and the one estimated by the robust regreassion is 81.63. We can verify that the robust model is performing well.
Note: “coef” was returned by the function “datasets.make_regression” when we created the data (see first code block).

### 3.3. Visualizing the fitted regression lines
```
lw = 2
plt.scatter(X[inlier_mask], y[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
plt.scatter(X[outlier_mask], y[outlier_mask], color='gold', marker='.', label='Outliers')
plt.plot(line_X, line_y, color='navy', linewidth=lw, label='Regural Linear regression')
plt.plot(line_X, line_y_ransac, color='royalblue', linewidth=lw, label='RANSAC regression')
plt.legend(loc='lower right')
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()
```

We observe again that the robust model is performing well ignoring the outliers.

## About me

**Piyush Pathak**

[**PORTFOLIO**](https://anirudhrapathak3.wixsite.com/piyush)

[**GITHUB**](https://github.com/piyushpathak03)

[**BLOG**](https://medium.com/@piyushpathak03)


# 📫 Follw me: 

[![Linkedin Badge](https://img.shields.io/badge/-PiyushPathak-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/piyushpathak03/)](https://www.linkedin.com/in/piyushpathak03/)

<p  align="right"><img height="100" src = "https://media.giphy.com/media/l3URDstnIjBNY7rwLB/giphy.gif"></p>
