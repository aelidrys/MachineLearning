# Linear Regression
## `I` - what is linear regression?
#### Linear Regression is a [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) algorithm in machine learning, which is widely used for solving regression problems. Regression is a type of machine learning problems where the goal is to predict a continuous output variable based on one or more input variables.

---
## `II` - Linear regression structer
### 1 - Line equation
#### Linear regression begins with the line equation `y = m*x + c`, which is simple. You have the parameters of the line, `m` and `c`, so you can take any input `x` and return the output `y`.
- #### `m` is the slope of the line (also called the weight),
- #### `c` is the intercept (the value of `y` when `x = 0`)

<img src="img/m_&_c.png" width=500>

### 2 - Linear regression in real problem

#### One of the most pobuler examples of linear regression usage, is house price prediction. Let's assume that we have a dataset that contains two columns: `house_size` as the input variable and `house_price` as the output variable.

#### The goal of linear regression is to learn the relationship between the input variable `house_size` and the output variable `house_price`, in order to predict the `house_price` of new houses based on their `house_size`.

#### For example if we have two examples:
- #### in the first the example `house_size = 4` and the `house_price = 2`.
- ####  in the second the example `house_size = 8` and the `house_price = 4`
#### We observe that the `house_price = house_size / 2` in this two previous examples. So, if we have a new example with `house_size = 6` we can predict that the `house_price = 3`

#### Let's take a example of a simple dataset:
| House Size | House Price |
|:----------:|:-----------:|
|     39     |     23      |
|     40     |     27      |
|    42.5    |    28.6     |
|     47     |     30      |
|     49     |     29      |
|     53     |     33      |
|     55     |   36.75     |
|     58     |     38      |
|     60     |    40.5     |
|     62     |    41.3     |

#### Dataset distribution graph.
<img src="img/data_distribution.png" width=500>


#### We observe that our data follows a linear distribution, so we can generate a line that represent the most points of our dataset.
<img src="img/data_best_line.png" width=500>

#### You can look at the graph to select the best fitting line among the three.

#### Visually, the green line appears to be the best fit. But how can we determine that computationally? How does a program judge which line is better than the others?
#### A program can judge which line is better than the others, by using cost function

### 3 - Cost or error function

#### The cost (or error) function calculates the average error for a given line. The error is the difference between the actual house price `y_r` and the predicted house price `y_p`.


#### `Cost = (y_p - y_r)²`
#### `y_p = m*x + c`
#### `cost = ((m*x+c)-y_r)²`

#### We use the exponent `²` to avoid negative values, because the difference `y_p - y_r` can be either positive or negative. Although we can use the absolute value, but squaring the error is generally preferred for several reasons.

#### A graph to simplify the cost
<img src="https://miro.medium.com/v2/resize:fit:1400/1*jmd_lPcwkZ6QByMfv2itXg.png" width="500">

#### To count the average value of error we aplly the cost function over all points of data, the exprission of average also named mean squared error  is shown in the image below


<img src="img/mse.png" width="500">

#### We compute the error in eash point and sume the errors. after that we divides the sume of errors over the number of points, finily we get the average value of error for our line

![img](img/error_table.png)

#### Now we can take a set of lines and calculate the cost (or average of error) for each one, and select the line with the minimum error.

#### However, this method doesn't work well in practice, because it requires generating a huge number of lines and hoping that one of them fits the data.

#### We need a way that is optimized and achievable.
#### There are two possible solutions:
- #### [Analytical method (Normal Equation)](#normal-equation)
- #### [Iterative method (Gradient Descent)](#gradient-descent)


### Gradient descent

#### Let's fix `c = -3.82` to observe how m impacts the fitting of the line.

<img src="img/lines_basedOn_m.png" width="1000">

#### When we increment `m`, the line fits the data more closely,  until it reaches the optimal fit at around `m = 0.72`. After this point, further increasing `m` makes the line fit the data less well.
- #### Before` m = 0.72`, increasing `m` reduces the error and improves the fit of the line.
- #### At `m = 0.72`, we achieve the optimal line with the best fit.
- #### After `m = 0.72`, increasing `m` increases the error, and the line fits the data less well.

<img src="img/cost_curve.png" width="600">

#### Once again, we require an approach that enables the program to find the minimum `cost` effectively.

#### We are dealing with a convex function and aim to find its minimum, which we can do using the derivative. From any point on the curve, we can move toward the minimum by following the gradient. To do this, we initialize the slope `m` with a random value, compute the derivative of the error function, and update `m` using the following rule: 
- #### `new_m = m - (gradient * learning_rate)`
- #### `gradient = d(error)/dm`

<img src="img/error_curve.png" width="600">


#### We repeat this process until the derivarive of the `error` ` d(error)/dm` becomes very small or equal to zero we have `new_m=m-(d(error)/dm*learning_rate)` => `new_m ≈ m` because `(d(error)/dm*learning_rate)` is equal or close to zero. The learning rate is a parameter that controls how large each update step is.

#### Finally, by following the above steps, we can find the best line that fits our data. This line is defined by its parameters `m` and `c`, and we can use it to predict new values. To make a prediction, we simply provide an input `x`, and compute: `y = mx + c` where `y` is the predicted output.

### Normal Equation
- #### not complete 

---
#### This is just a simple example of the theory behind linear regression. For the implementation, you can check [here](42ft_linear_reg/README.md)


