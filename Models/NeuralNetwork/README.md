## Not complete in progress!!!
# Neural Network
#### To follow up in neural network is recommended to take a view about [Linear Regression](../LinearRegression/README.md)

### what is neural network
##### A neural network is a machine learning program, or model, that makes decisions in a manner similar to the human brain, by using processes that mimic the way biological neurons work together to identify phenomena, weigh options and arrive at conclusions. 

<img src="imgs/brain_neurens.jpg" width=800px>

## Prerequests
- #### Linear Regression [MyDocs](../LinearRegression/README.md)
- #### Chain Rule derivative
  - ![](imgs/chain-rule.jpg)
  - ![](imgs/chain-rule-extended.jpg)
- #### Graph Theory
  - <img src="imgs/graph_threory.png" width=400>
---
## Why Neural Network
### Neural Network come to resolve the limitations of teaditional Regression models
- #### Regression models like linear regression and logistic regression are not capable of learning non-linear patterns, complex interactions, or hierarchical representations.
- #### Regression models can use techniques like polynomial features to capture non-linear patterns, but this often leads to increased computational time and model complexity.
- #### Regression models struggle with High-Dimensional Data Neural Network can handle high-dimensional input, such as images, audio, or sequences.
---

## Neural Network Workflow
#### Neural Network start from a linear regression model
#### lets represent the Linear Regresion as a graph
<img src="imgs/LR_graph.png" width=500>

#### Neural networks start from the foundation of linear models like linear regression, and extend them by adding hidden layers and non-linear activation functions, allowing them to learn more complex patterns.

#### In the graph below, we extend the linear regression model by adding a hidden layer in the middle. Since we do not apply any activation function.
<img src="imgs/nn1.png" width=800>

#### the overall transformation remains linear:
#### `O` = `w5(x1*w1+w2*x2+b1) + w6(x1*w3+w4*x2+b2)` = `X1*w1*w5 + x2*w2*w5 + b1*w5 + ...`
#### for example `w1*w5` its not non-linear transformation and we can summarize `w1*w5` in `w` just a single `w` can do the same effect so it still a linear regression model

<img src="imgs/nn2.png" width=900>

- [ ] practice examlple to be familair with feed forword 
- [ ] Move to Backprobagation