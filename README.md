<p align="center">
<img src="https://i1.wp.com/theconstructor.org/wp-content/uploads/2014/04/compression-test-on-cylinder.JPG-1-1.jpg?fit=712%2C477&ssl=1" width="500" />

# Concrete-compressive-strength

The Compressive Strength of Concrete determines the quality of concrete. The compression strength of concrete is a measure of the concrete's ability to resist loads which tend to compress it. It is measured by crushing cylindrical concrete specimens in compression testing machine. Concrete is the most important material in building construction, and its compressive strength is a non-linear function of its age and components. In this repo, we estimate this problem through linear regrtession techniques as a function of its characteristics.

# Dataset 

The Concrete Compressive Strength DataSet consists of 1030 observations under 9 attributes. There are 8 input variables and 1 output variable. Seven input variables represent the amount of raw material (measured in kg/m³) and one represents Age (in Days). The target variable is Concrete Compressive Strength measured in (MPa). The attributes include factors that affect concrete strength such as cement, water, aggregate (coarse and fine), and fly ash etc... Also, this dataset is obtained from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/concrete+compressive+strength).
  
* Number of instances - 1030
* Number of Attributes - 9
  * Attribute breakdown - 8 quantitative inputs, 1 quantitative output
  
  
<div align="center">
  
| Attributes | Unit | 
| :---: | :---: | 
| Cement | kg/m³ | 
| Blast Furnace Slag | kg/m³ | 
| Fly Ash | kg/m³  | 
| Water | kg/m³ | 
| Superplasticizer | kg/m³ | 
| Coarse Aggregate | kg/m³ | 
| Fine Aggregate| kg/m³ | 
| Age | Days | 
| Concrete Compressive Strength | MPa | 

</div>
<figcaption align = "center"><p align="center">Table 1. The features of the Concrete Compressive Strength DataSet.</figcaption>
</figure>
 
  
## Modelling and Evaluation
**ALGORITHMS**

* *Linear regression*
* *Lasso regression*
* *Ridge regression*

**METRICS**

Since the target variable is a continuous variable, regression evaluation metric MSE (Mean Squared Error), MAE (Mean Absolute Error) and MAPE Score (Mean Absolute Percentage Error) have been used.


<div align="center">

![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20MSE%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Cleft%20%5C%7Cy-%5Chat%7By%7D%20%5Cright%20%5C%7C%5E2)
  
![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20MAE%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Cleft%20%5C%7Cy-%5Chat%7By%7D%20%5Cright%20%5C%7C)
  
![equation](https://latex.codecogs.com/gif.latex?%5Cbg_white%20MAPE%20%3D%20%5Cfrac%7B1%7D%7BN%7D%5Csum_%7Bi%3D1%7D%5EN%5Cleft%20%7C%20%5Cfrac%7By%28i%29-%5Chat%20y%28i%29%7D%7By%28i%29%29%7D%20%5Cright%20%7C)
</div>



## Exploratory Data Analysis

The first step is to understand the data and gain insights from the data before doing any modelling. This includes checking for any missing values, plotting the features with respect to the target variable, observing the distributions of all the features and so on. 
```In Figure 1```, we display the correlation between the features through heatmap and in ```Figure 2``` the pairplot in seaborn to plot pairwise relations between all the features and distributions of features along the diagonal.

<p align="center">
<img src="https://user-images.githubusercontent.com/80779522/139156716-1f23371a-23f6-4ccc-baae-a49912e37608.png" width="600" />
<figcaption align = "center"><p align="center">
  Figure 1. Correlation betweem features.</figcaption>
</figure>


<p align="center">
<img src="https://user-images.githubusercontent.com/80779522/139157501-32bc9a31-7e2c-4210-8962-b4c177d7993b.png" width="850" />
<figcaption align = "center"><p align="center">
  Figure 2. Visual representation of correlations (pairplot).</figcaption>
</figure>

**METHODOLOGY**

*STEP N<sup>o1</sup>*

The linear algorithms are tested with different values of the ```alpha``` parameter. In specific:

*alpha* = [10<sup>-3</sup>, 10<sup>-2</sup>, 10<sup>-1</sup>, 1 , 5, 10, 10<sup>2</sup>, 10<sup>3</sup>]

As the value of the alpha parameter increases, the complexity of the Ridge model increases in both training and evaluation process, while the complexity of Lasso model in the prediction process remains constant. This is because the Ridge model takes account into all the features of the dataset, while the Lasso model performs feature selection, and in particular, the coefficients of the other features are zeroed or reduced by a fixed factor.  

*STEP N<sup>o2</sup>*

In order to select the optimal value of the alpha parameter, cross validation method was applied. In the case of linear regression the model parameters are independent of each other, then the logarithmic probability function of the model parameter vector represents the contribution of each parameter. Thus, by changing the alpha values, we basically control the coefficients penalty. The higher its values, the higher the penalty and therefore the smaller the values of the feature coefficients. 

The dataset was randomly splitted into 70% for ```training``` and 30% for ```testing```. This process is repeated 10 times and as a result the *__average__* and the *__standard deviation__* of each metric was calculated.

*STEP N<sup>o3</sup>*

Given the non-linearity of the function we are trying to model, it is worth evaluating more expressive linear regression models with ```polynomial``` terms of the features. For this reason, a function 

```
test_poly_regression(X_train, y_train, X_test, y_test, n = 2)
```
was implemented. In specific:

* Inputs
  * X_train: training set
  * y_train: labels of the training set
  * X_test: testint set
  * y_test: labels of the testing set
  * n: degree of polynomial 𝑛≥1

* Outputs
  *  A new set of features consisting of the original features and their versions elevated to powers up to 𝑛.

## Dependencies 
Install all the neccecary dependencies using ```pip3 install <package name>```
  
Required packages:
  ```
  - numpy (Version >= 1.19.4)
  - matplotlib (Version >= 3.4.3)
  - scikit-learn (Version >= 0.22.2)
  - seaborn (Version >= 0.10.1)
  - pandas (Version >= 1.0.3)
```

  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
