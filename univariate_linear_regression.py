import numpy as np

# calculating predections for (w,b)
def f_x(x,w,b):
    return x * w + b

# computing cost at (w,b)
def j_wb(x, y, w, b, Lambda = 0):
    m = x.shape[0]
    return  (np.sum((f_x(x,w,b) - y)**2)/(2*m) + (Lambda*(w**2))/(2*m))

def dj_dw(x,y, w,b, Lambda = 0):
    m = x.shape[0]
    return (np.sum((f_x(x,w,b) - y)*x)/m+ (Lambda*w)/m)

def dj_db(x,y, w,b):
    m = x.shape[0]
    return np.sum((f_x(x,w,b) - y))/m


def alpha_backtracking(x, y, w, b, alpha, Lambda = 0):
    j_old = j_wb(x,y,w,b,Lambda)
    w_grad  =dj_dw(x,y, w, b, Lambda)
    b_grad  =dj_db(x,y, w, b)
    # while True:
    #     w_temp = w - alpha * w_grad
    #     b_temp = b - alpha * b_grad
    #     j_new = j_wb(x,y,w_temp,b_temp, Lambda)
    #     if j_new > j_old:
    #         break
    #     alpha *= 1.03
    # alpha /= 1.03
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

def gradient_descent(x,y,w,b, num_iters, alpha = None, Lambda = 0):
    if not alpha:
        alpha = 1/np.max(x)**2
    parameters_history = np.array([w,b]).reshape(1,2)
    j_history = np.array([j_wb(x,y,w,b, Lambda)])
    j_mag_prev = 0
    for i in range(num_iters):
        alpha = alpha_backtracking(x,y,w,b,alpha, Lambda)
        djdw = dj_dw(x,y,w,b, Lambda)
        djdb = dj_db(x,y,w,b)
        j_mag = np.sqrt(djdw**2 + djdb**2)
        if j_mag < 1e-8 or j_mag == j_mag_prev:
            return w,b,parameters_history, j_history
        j_mag_prev = j_mag
        w_temp = w- alpha * djdw
        b_temp = b- alpha * djdb
        w,b = w_temp, b_temp
        parameters_history = np.append(parameters_history, np.array([w,b]).reshape(1,2), axis= 0)
        j_history = np.append(j_history, j_wb(x,y,w,b,Lambda))
        if i % int(np.ceil(num_iters/1000)) == 0:
            print(f"iteration no {i}, J(w,b) = {j_history[-1]}, Parameters (w,b) = ({w},{b}), J_mag = {j_mag}, learning rate = {alpha}")
    return w,b,parameters_history, j_history