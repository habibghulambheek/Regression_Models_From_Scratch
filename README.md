
# Regression Models From Scratch  

This repository contains implementations of different regression algorithms **from scratch using NumPy**, without relying on machine learning libraries like scikit-learn.  

## 📌 Implemented Models  
1. **Univariate Linear Regression** – one feature, straight-line fitting.  
2. **Multiple Linear Regression** – multiple features, extended linear model.  
3. **Logistic Regression** – binary classification using sigmoid and cross-entropy loss.

## 🛠 Features  
- Gradient Descent optimization (with stopping criteria).  
- Cost function and gradient calculations derived step by step.  
- **Regularization (Ridge penalty)** in linear & logistic regression.  
- Training history tracking (weights, bias, and cost).  
- Lightweight and fully vectorized using **NumPy**.  

## 📂 Project Structure  
```

Regression\_Models\_From\_Scratch/
│── univariate\_linear.py
│── multiple\_linear.py
│── logistic.py
│── README.md

````

## 🚀 Usage  
Clone the repo:  
```bash
git clone https://github.com/habibghulambheek/Regression_Models_From_Scratch.git
cd Regression_Models_From_Scratch
````

Run an example (e.g., logistic regression):

```python
from LogisticRegression import *
from matplotlib import pyplot

# Sample Dataset
x =  np.array([0,1,1.5,3,4,4.5,5,4.3,6]).reshape(-1,1)
y = np.array([0,0,0,0,1,1,1,1,1])

# Intialize
w = np.array([0])
b = 0
alph = 0.5
iters = 10000
Lambda = 0
# train
w_out, b_out,j_hist,w_hist,b_hist= gradient_descent(x, y, w, b, alph,Lambda, iters) 
x0 = -b_out/w_out[0] # Boundary


# Plot Dicision Boundary & Model
pos = y==1
neg = y==0
pyplot.scatter(x[pos],y[pos],marker='x', c = 'r', label = "p(y) = 1")
pyplot.scatter(x[neg],y[neg],marker='o', c = 'b', label = "p(y) = 0")
pyplot.plot(x,f_wb(z(x,w_out,b_out)))
pyplot.axvline(x0, color='g', label='Decision Boundary')  
pyplot.show()

```


## 🙌 Acknowledgements

Inspired by **Andrew Ng’s Machine Learning Specialization** and implemented step by step for deeper understanding.




