To calculate the straight regression line from fictive measurement points (called engine retardation/linear regression), a floating Linear Least Squares Fit (LLSF) algorithm is used. The LLSF estimation is a good method if assumptions are met to obtain regression weights when analyzing the engine data. However, if the data does not satisfy some of these assumptions, then sample estimates and results can be misleading. Especially, outliers violate the assumption of normally distributed residuals in the least-squares regression. The fact of outlying engine power data points (engine dips), in both the direction of the dependent (y-axis) and independent variables (x-axis/timestamp), to the least-squares regression is that they can have a strong adverse effect on the estimate and they may remain unnoticed. Therefore, techniques like RANSAC (Random Sample Consensus) that are able to cope with these problems or to detect outliers (bad) and inliers (good) have been developed by scientists and implemented into SimplexNumerica.


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

## 1. HUBER Regression
## 2. RANSAC Regression

### 2.1 RANSAC Regression
Random Sample Consensus (RANSAC) is a well-known robust regression algorithm 
RANSAC tries to separate data into outliers and inliers and fits the model only on the inliers.
In this article we will only use RANSAC but almost all statements are true for the Huber Robust regression as well.

The Random Sample Consensus (RANSAC) algorithm proposed by Fischler and Bolles[3] is a general parameter estimation approach designed to cope with a large proportion of outliers in the input data. Its basic operations are:

1. Select sample set
2. Compute model
3. Compute and count inliers
4. Repeat until sufficiently confident

The RANSAC steps in more details are[4]:

1. Select randomly the minimum number of points required to determine the model parameters.
2. Solve for the parameters of the model.
3. Determine how many points from the set of all points fit with a predefined tolerance.
4. If the fraction of the number of inliers over the total number of points in the set exceeds a predefined threshold, re-estimate the model parameters using all the identified inliers and terminate.
5. Otherwise, repeat steps 1 through 4 (maximum of N times).

Briefly, RANSAC uniformly at random selects a subset of data samples and uses it to estimate model parameters. Then it determines the samples that are within an error tolerance of the generated model.

These samples are considered as agreed with the generated model and called as consensus set of the chosen data samples. Here, the data samples in the consensus as behaved as inliers and the rest as outliers by RANSAC. If the count of the samples in the consensus is high enough, it trains the final model of the consensus by using them.

It repeats this process for a number of iterations and returns the model that has the smallest average error among the generated models. As a randomized algorithm, RANSAC does not guarantee to find the optimal parametric model with respect to the inliers. However, the probability of reaching the optimal solution can be kept over a lower bound by assigning suitable values to algorithm parameters.

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

#### The first is a new robust estimator MLESAC that is a generalization of the RANSAC estimator. It adopts the same sampling strategy as RANSAC to generate putative solutions but chooses the solution that maximizes the likelihood rather than just the number of inliers. The second part of the algorithm is a general-purpose method for automatically parameterizing these relations, using the output of MLESAC.

## Maximum Likelihood Estimator Sample Consensus (MLESAC)

In particular, MLESAC is well suited to estimating the Engine Retardation trend or more general, it manifolds the engine’s power data to timestamp miss relation in Engine Retardation measurement because of the fact that the timestamp is set maybe inaccurately inside the internal clock of the measurement device.

Technical descriptions and own tests have shown that the RANSAC algorithm has been proven very successful for robust estimation, but with the robust negative log-likelihood function having been defined as the quantity to be minimized it becomes apparent that RANSAC can be improved on. One of the problems with RANSAC is that if the threshold for considering inliers is set too high then the robust estimate can be very poor and the slope of the regression line goes wrong.

As an improvement over RANSAC, MLESAC has a better estimate for the elimination of noise dips for instance influenced by neighborhood machines. The minimal set point, initially selected by MLESAC, is known to provide a good estimate of the data relation. Hence, the initial estimate of the point basis provided by MLESAC is quite close to the true solution and consequently, the nonlinear minimization typically avoids local minima. Then the parameterization of the algorithm is consistent, which means that during the gradient descent phase-only data relations that might actually arise are searched for. It has been observed that the MLESAC method of robust fitting is good for initializing the parameter estimation when the data are corrupted by outliers. In this case, there are just two classes to which a datum might belong, inliers or outliers.

Torr and Zisserman have shown that the implementation of MLESAC yields a modest to hefty benefit to all robust estimations with absolutely no additional computational burden. In addition, the definition of the maximum likelihood error allows it to suggest a further improvement against RANSAC. As the aim is to minimize the negative log-likelihood of the data it makes sense to use this as the score for each of the random samples.

After MLESAC is applied, nonlinear minimization is conducted using the method described in Gill and Murray[6], which is a modification of the Gauss-Newton method. All the points are included in the minimization, but the effect of outliers is removed as the robust function places a ceiling on the value of their errors, unless the parameters move during the iterated search to a value where that correspondence might be reclassified as an inliers. This scheme allows outliers to be reclassed as inliers during the minimization itself without incurring additional computational complexity. This has the advantage of reducing the number of false classifications, which might arise by classifying the correspondences at too early a stage.

 

Evaluation of Samples
To show some results of the new SimplexNumerica algorithms, the following samples are evaluated. All have simulated data randomized around the slope f(x) = m x + b, m = 1/36, b = 1000. The inverse value of the difference quotient (m) is equal to the rundown time in (s/W). The next figure shows two outliers down under the theoretical graph – fitted by RANSAC (green line).

Example with two outliers:



## For more details ,refer this link
https://www.coursera.org/lecture/robotics-perception/ransac-random-sample-consensus-i-z0GWq

## About me

**Piyush Pathak**

[**PORTFOLIO**](https://anirudhrapathak3.wixsite.com/piyush)

[**GITHUB**](https://github.com/piyushpathak03)

[**BLOG**](https://medium.com/@piyushpathak03)


# 📫 Follw me: 

[![Linkedin Badge](https://img.shields.io/badge/-PiyushPathak-blue?style=flat-square&logo=Linkedin&logoColor=white&link=https://www.linkedin.com/in/piyushpathak03/)](https://www.linkedin.com/in/piyushpathak03/)

<p  align="right"><img height="100" src = "https://media.giphy.com/media/l3URDstnIjBNY7rwLB/giphy.gif"></p>
