import numpy as np
def f_wb(z):
    return 1/(1+ np.exp(-z))

def z(x,w,b):
    return x.dot(w) + b 


def j_wb(x,y,w,b,Lambda = 0,eplison =  1e-15):
    m = y.shape[0]
    f = f_wb(z(x,w,b))
    cost = -(np.sum(y*np.log(f+eplison)+ (1-y)* np.log(1-f+eplison)))/m
    reg_cost  =  (Lambda/(2*m))*np.sum(w**2)
    return cost + reg_cost

def compute_gradients(x,y,w,b, Lambda = 0):
    dj_dw = 0
    dj_db = 0
    m = x.shape[0]
    z_tmp = z(x,w,b)
    error =  (f_wb(z_tmp) - y)
    dj_dw = np.sum((x * error.reshape(m,-1)),axis = 0 )/m + (Lambda/m)*w
    dj_db = np.sum(error)/m
    return dj_dw, dj_db 
def gradient_descent(x,y,w,b, alpha,Lambda = 0, num_iter= 1000, cost_diff = 1e-8, min_cost = 1e-8, parmeters_diff = 1e-8):
    """return w_tmp,b, np.array(j_hist),np.array(w_hist),np.array(b_hist)
"""
    w_tmp =  w.copy()
    report_every = int(np.ceil(num_iter/10))

    curr_cost = j_wb(x,y,w,b,Lambda)   
    prev_cost = None
    j_hist  = [curr_cost]
    w_hist = [w]
    b_hist = [b]

    for  i in range(num_iter):
        dj_dw, dj_db= compute_gradients(x,y,w_tmp,b,Lambda)        
        w_tmp = w_tmp -  alpha * dj_dw
        b =  b -  alpha * dj_db
        
        if curr_cost < min_cost or (np.allclose(w_tmp , w_hist[-1],atol=parmeters_diff, rtol=0) and b == b_hist[-1]):
            return w_tmp,b, np.array(j_hist),np.array(w_hist),np.array(b_hist)

        prev_cost = curr_cost
        curr_cost = j_wb(x,y,w_tmp,b,Lambda) 

        if abs(curr_cost - prev_cost) < cost_diff:
            return w_tmp,b, np.array(j_hist),np.array(w_hist),np.array(b_hist)

        
        if i < 10000:
            j_hist.append(curr_cost)
            w_hist.append(w_tmp.copy())
            b_hist.append(b)
        if(i%report_every== 0):
            print(f"i = {i}, w  =  {w_tmp}, b = {b}, Cost function = {j_hist[-1]} .")

    return w_tmp,b, np.array(j_hist),np.array(w_hist),np.array(b_hist)
