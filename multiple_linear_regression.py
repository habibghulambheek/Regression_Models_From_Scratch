import numpy as np

def f_wb(x,w ,b):
    return x.dot(w) + b
def j_wb(x,y,w,b, Lambda = 0):
    m = x.shape[0]
    return (np.sum((f_wb(x,w,b)-y)**2)/(2*m)  + (Lambda/(2*m)) * np.sum(w**2))

def compute_gradients(x,y,w,b, Lambda = 0):

    m = x.shape[0]
    n = x.shape[1]
    w_grad = np.zeros(n)    
    error = f_wb(x,w,b) - y

    w_grad = ((error[:, None] * x).sum(axis = 0)/m + (Lambda/m)*w)
    b_grad = np.sum(error)/m
    return w_grad, b_grad


def alpha_backtracking(x, y, w, b, alpha, Lambda):
    j_old = j_wb(x,y,w,b,Lambda)
    w_grad ,b_grad = compute_gradients(x,y,w,b, Lambda)
    while True:
        w_temp = w - alpha * w_grad
        b_temp = b - alpha * b_grad
        j_new = j_wb(x,y,w_temp,b_temp, Lambda)
        if j_new < j_old:
            break
        alpha *= 0.97

        if alpha < 1e-6:
            break
    return alpha

  
def gradient_descent(x,y,w_in,b, num_iter,alpha,Lambda = 0, min_cost = 1e-6, min_grad = 1e-8, min_diff = 1e-6):
    """
    x: features
    y: target values
    w_in: initial weights
    b: initial bias
    num_iter: number of iterations
    alpha: Initial learning rate
    min_cost: Minimum cost (1e-6)
    min_grad: minimum magnitude of gradient (1e-8)
    min_diff: minimum difference between new and previous costs (1e-6)
    return type: (w,b,j_hist,w_hist,b_hist)
    """
    w =  w_in.copy()
    n = w.shape[0]
    j_hist = [j_wb(x,y,w,b,Lambda)]
    b_hist = [b]
    w_hist = [w]
    report_every = int(np.ceil(num_iter/10))
    new_grad = j_hist[-1]
    old_grad = None
    for i in range(num_iter):
        
        alpha = alpha_backtracking(x,y,w,b,alpha, Lambda)
        w_grad,b_grad = compute_gradients(x,y,w,b, Lambda)
      

        j_mag = np.sqrt(np.dot(w_grad, w_grad) + b_grad**2)

        if new_grad <= min_cost  or (old_grad != None and abs(new_grad- old_grad)<= min_diff)  or j_mag <= min_grad:
            # print((old_grad ,new_grad, abs(new_grad- old_grad), min_diff))
            return w,b,np.array(j_hist),np.array(w_hist),np.array(b_hist)
        

        w -= alpha*w_grad
        b -= alpha*b_grad
        old_grad = new_grad
        new_grad = j_wb(x,y,w,b, Lambda)
        if i < 100000: # preventing resourse exhastation
            j_hist.append(new_grad)
            b_hist.append(b)
            w_hist.append(w)

        if i % report_every==0:
             print(f"iteration no {i}, J(w,b) = {j_hist[-1]}, J_mag = {j_mag}, learning rate = {alpha}")   
    return w,b,np.array(j_hist),np.array(w_hist),np.array(b_hist)