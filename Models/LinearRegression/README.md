# Linear Regression
## what is linear regression?
#### Linear Regression is a [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning) algorithm in machine learning, which is widely used for solving regression problems. Regression is a type of machine learning problems where the goal is to predict a continuous output variable based on one or more input variables.


### example of linear regression usage
##### the popular example of linear regression usage is house price prediction

<img src="https://miro.medium.com/v2/resize:fit:1024/0*YMZOAO8QE4bZ4_Rk.jpg" width="500">


#### assume that there is a linear relationship between size of house and his price.
#### the goal of linear regression is to find the best fitting line to describe the relationship between the input (house size) and the output (house price).

<img src="img/linear_reg.png" width="800">

#### in figure 1 we have alotof lines. each line represented by m and c.
#### m the slope of line defined by `m =  (y₂ - y₁) / (x₂ - x₁)` and c the intercept of the line with y axis show figure 2
#### each line allows us to predict the price of each house, denoted by `Y`, based on its size, denoted by `X`. by using the equation of the line `y = m*x + c`, but the predicted price may not be valid depends on the ability of this line to discribe the data.

#### In the figure 1 there are different lines and the best line is the orange line. We choose this line by seeing the figure and comparing the lines, but the computer cannot do this, there are a set of complex operations that the computer performs to choose the best line.


#### we define a error function or cost function that count the average value of error for each line, the error is the difference between the real house price `y_r` and the predicted house price `y_p`


#### `error = (y_p - y_r)^2`
#### `y_p = m*x + c`
#### `error = ((m*x+c)-y_r)^2`

#### We use the exponent `^2` to avoid a negative value because sometimes the difference `y_p - y_r` is positive and sometimes it is negative, 

<img src="https://miro.medium.com/v2/resize:fit:1400/1*jmd_lPcwkZ6QByMfv2itXg.png" width="500">

#### to count the average value of error we aplly this error function over all points of data, the exprission of average also named mean squared error  is shown in the image below


<img src="img/mse.png" width="500">

#### we compute the error in eash point and sume the errors. after that we divides the sume of errors over the number of points, finily we get the average value of error for our line

![img](img/error_table.png)

#### now we can take a set of lines and calculate the error (average of error) in each one of them, and select the line that have minimum error but this method not work, because we need to generate a huge number of lines and hupfuly there is one line that can fit the data

#### We know that there is one line that gives the minimum error, so we can use the graph of the error curve as a function of the slope `m` to make this clearer, The `x-axis` represents `m`, and the `y-axis` represents the error. We consider the intercept `c` as a constant to simplify the analysis.

![](img/error_curve.png)

#### We are dealing with a convex function, and we want to find its minimum using the derivative. From any point on the curve, we can move toward the minimum by following the gradient. To do this, we initialize the slope m with a random value, compute the derivative of the error function, and update m using the rule: 
#### `new_m = m - (d(error)/dm * learning_rate)`

#### We repeat this process until the derivarive of the error d(error)/dm becomes very small or equal to zero we have `new_m=m-(d(error)/dm*learning_rate)` => new_m=m because `(d(error)/dm*learning_rate)` is equal or close to zero. The learning rate is a parameter that controls how large each update step is.

#### Finally, by following the above steps, we can find the best line that fits our data. This line is defined by its parameters `m` and `c`, and we can use it to predict new values. To make a prediction, we simply provide an input `x`, and compute: `y = mx + c` where `y` is the predicted output.

#### This is just a simple example of the theory behind linear regression. For the implementation, you can check [here](42ft_linear_reg/Document.md)


