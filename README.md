
# Regression Models From Scratch  

This repository contains implementations of different regression algorithms **from scratch using NumPy**, without relying on machine learning libraries like scikit-learn.  

## 📌 Implemented Models  
1. **Univariate Linear Regression** – one feature, straight-line fitting.  
2. **Multiple Linear Regression** – multiple features, extended linear model.  
3. **Logistic Regression** – binary classification using sigmoid and cross-entropy loss, with **L2 regularization** support.  

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
from logistic_regression.logistic import gradient_descent, predict, accuracy
import numpy as np

# sample dataset
X = np.array([[1,2],[2,3],[3,4],[4,5]])
y = np.array([0,0,1,1])

# initialize
w = np.zeros(X.shape[1])
b = 0

# train
w, b, j_hist, _, _ = gradient_descent(X, y, w, b, alpha=0.1, num_iter=1000, lambd=0.1)

print("Final Weights:", w)
print("Final Bias:", b)
print("Accuracy:", accuracy(X,y,w,b))
```


## 🙌 Acknowledgements

Inspired by **Andrew Ng’s Machine Learning Specialization** and implemented step by step for deeper understanding.




