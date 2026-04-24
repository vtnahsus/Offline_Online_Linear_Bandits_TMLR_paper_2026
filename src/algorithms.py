import numpy as np
import scipy as sp
from src.utils import sample_arm, compute_d_e

# Helper function for OOPE.
# O(d) initialization function
# Input: A_l-live arm indices, d_ dimension, A- arm matrix
# Output: numpy array of arm indices.
def O_d_initialization(A_l,d,A):
  c=np.zeros(d)
  c[0]=1
  B=[]
  A_l_ids=[]
  for i in range(d):
    max_arm_id=np.argmax(np.absolute(np.matmul(A[A_l],c)))                         #Choose the arm which maximizes the dot product wrt c.
    A_l_ids.append(max_arm_id)
    if(i==0):
      B.append(A[A_l][max_arm_id]/np.linalg.norm(A[A_l][max_arm_id]))
    else:
      t=A[A_l][max_arm_id]-np.dot(np.array(B),A[A_l][max_arm_id])@np.array(B)      #Do a Grant-Schmidt orthognalization of (B). This helps in creating the next c.
      B.append(t/np.linalg.norm(t))

    eps=1
    while(eps>1e-10):
      c=np.random.randn(d)
      c=c-np.matmul(np.dot(np.array(B),c),np.array(B))                               #Choose a vector c orthogonal to span(B).
      c=c/np.linalg.norm(c)
      eps=np.amin(np.dot(np.array(B),c))
    #if(np.amin(np.dot(np.array(B),c))>1e-10):
    #  raise ValueError('Orthognalization is not there in the initialization.')
  return(np.array(A_l_ids))

def opt_log_det_sp(alpha,A_l,d,A,V_pi_o,online_frac=np.array([]),test=False):
  if(len(online_frac)==0):
    online_frac=np.ones(len(A_l))/len(A_l)
  online_frac_dict=dict.fromkeys(A_l,0)
  if(alpha==0 & len(A_l)<=d):
    for i in range(len(A_l)):
      online_frac_dict[A_l[i]]=online_frac[i]
    return(online_frac_dict)

  def objective_function(w):
    s=alpha*V_pi_o
    i=0
    for a in A_l:
      s=s+(1-alpha)*w[i]*(np.outer(A[a],A[a]))
      i=i+1
    return(-np.log(np.linalg.det(s)))

  def gradient(w):
    s=alpha*V_pi_o
    i=0
    for a in A_l:
      s=s+(1-alpha)*w[i]*(np.outer(A[a],A[a]))
      i=i+1
    H=np.linalg.pinv((1-alpha)*s+alpha*V_pi_o)
    grad=(1-alpha)*np.linalg.norm(np.matmul(A[A_l], sp.linalg.sqrtm(H)), axis=1)
    return(grad)


  constraints=[{'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
               {'type': 'ineq', 'fun': lambda x: x}]
  initial_guess=online_frac
  bounds = [(0, 1)]*len(A_l)
  result = sp.optimize.minimize(objective_function, initial_guess,jac=gradient, bounds=bounds,constraints=constraints,method="SLSQP",options={'ftol':1e-12,'eps':1e-12,'maxiter':1e5})

  online_frac=result.x
  #online_frac = np.clip(online_frac, 0, 1)  # Clip values to be within [0, 1]
  online_frac/= np.sum(online_frac)
  if(test):
    print(result)
  for i in range(len(A_l)):
    online_frac_dict[A_l[i]]=online_frac[i]
  return(online_frac_dict)

# Helper function for OOPE.
# Frank-Wolfe implementation.
# Input: alpha- offline fraction, A_l-set of live arms, T-horizon,d-dimension, A- arm matrix, V_pi_o- offline gram matrix.
# Output: Approximate optimal solution in dictionary form- keys are the arm indices and data is arm pull fraction.
def Frank_Wolfe(T_o,T,alpha,A_l,d,A,V_pi_o,test=False):
  if(len(A_l)>d):
    init_arm_ids=O_d_initialization(A_l,d,A)
    m=d
  else:
    init_arm_ids=np.arange(len(A_l)).astype(int)
    m=len(A_l)
  online_frac=np.zeros(len(A_l))
  online_frac_dict=dict.fromkeys(A_l,0)
  for i in init_arm_ids:
    online_frac[i]=1/m
  if(alpha==0 & len(A_l)<=d):
    for i in range(len(A_l)):
      online_frac_dict[A_l[i]]=online_frac[i]
    return(online_frac_dict)
  V_pi_n=np.matmul(A[A_l].T,online_frac[:,None]*A[A_l])
  H=np.linalg.pinv((1-alpha)*V_pi_n+alpha*V_pi_o)
  w=(1-alpha)*np.linalg.norm(np.matmul(A[A_l], sp.linalg.sqrtm(H)),axis=1) + alpha*np.trace(np.matmul(H, V_pi_o))
  a_plus=np.argmax(w)
  delta=np.amax(w)/d-1
  #print(delta)
  d_eff=compute_d_e(T_o,T,V_pi_o)
  eps=d_eff/d #0.9*np.power(d/T,1/3)
  while(delta>eps):
    beta=max((w[a_plus]-d)/((d-1)*w[a_plus]),-online_frac[a_plus])
    online_frac[a_plus]=online_frac[a_plus]+beta
    online_frac=(1/(1+beta))*online_frac
    V_pi_n=(1/(1+beta))*V_pi_n+(beta/(1+beta))*np.outer(A[A_l[a_plus]],A[A_l[a_plus]])
    H=np.linalg.inv((1-alpha)*V_pi_n+alpha*V_pi_o)
    w = (1-alpha)*np.linalg.norm(np.matmul(A[A_l], sp.linalg.sqrtm(H)),axis=1) + alpha*np.trace(np.matmul(H, V_pi_o))
    a_plus=np.argmax(w)
    delta=np.amax(w)/d-1
  if(test):
    print(delta)
  for i in range(len(A_l)):
    online_frac_dict[A_l[i]]=online_frac[i]
  return(online_frac_dict)

def Eliminate(A_l,hat_theta,A,e_l):
  max=float('-inf')
  for i in range(len(A_l)):
    if(max<np.dot(hat_theta,A[A_l[i]])):
      max_id=A_l[i]
      max=np.dot(hat_theta,A[A_l[i]])
  #print("max id",max_id)
  #print("diagnostic",np.dot(A[max_id]-A[optimal_arm],hat_theta)-2*e_l)
  temp=[]
  for a in A_l:
    if(max-np.dot(hat_theta,A[a])<2*e_l):
      temp.append(a)
  return(np.array(temp).astype(int))


# --- ALGORITHMS ---

def OOPE(T, A, d, non_zero_arm, T_o, optimal_arm, theta, V_pi_o, offline_frac_dict, offline_data_dict, use_fw=False):
    l = 1
    A_l = np.arange(len(A))
    s = 0
    regret = 0
    offline_indices = np.zeros(len(non_zero_arm)).astype(int)
    offline_left = True

    while s < T + 1:
        if len(A_l) > 1:
            e_l = np.power(0.5, l)
            alpha_l = T_o / (T + T_o) if offline_left and T_o > 0 else 0
            
            if use_fw:
                online_frac_dict = Frank_Wolfe(T_o, T, alpha_l, A_l, d, A, V_pi_o)
            else:
                online_frac_dict = opt_log_det_sp(alpha_l, A_l, d, A, V_pi_o)
                
            keys = np.array(list(online_frac_dict.keys()))
            V_pi_n = np.zeros(shape=(d, d))
            for a in keys:
                V_pi_n += online_frac_dict[a] * np.outer(A[a], A[a])
                
            H = np.linalg.pinv((1 - alpha_l) * V_pi_n + alpha_l * V_pi_o)
            g_pi = np.amax(np.linalg.norm(np.matmul(A[A_l], sp.linalg.sqrtm(H)), axis=1))**2
            
            i = 0
            start = True
            offline_samples_avail = False
            for a in non_zero_arm:
                n_offline_a = np.ceil(2 * alpha_l * offline_frac_dict[a] * g_pi * np.log(4 * np.power(l, 2) * len(A) * T) / (np.power(e_l, 2))).astype(int)
                start_pos = offline_indices[i]
                end_pos = min(offline_indices[i] + n_offline_a, len(offline_data_dict.get(a,[])))
                
                if end_pos > start_pos:
                    offline_samples_avail = True
                    temp_array = offline_data_dict[a][start_pos:end_pos]
                    n_samples = end_pos - start_pos
                    if start:
                        design_matrix = np.tile(A[a], [n_samples, 1])
                        y_vec = temp_array
                        start = False
                    else:
                        design_matrix = np.vstack((design_matrix, np.tile(A[a], [n_samples, 1])))
                        y_vec = np.hstack((y_vec, temp_array))
                    offline_indices[i] = end_pos
                else:
                    offline_left = False
                i += 1
                
            j = 0
            for a in A_l:
                n_online_a = np.ceil(2 * (1 - alpha_l) * g_pi * online_frac_dict[a] * np.log(4 * np.power(l, 2) * len(A) * T) / (np.power(e_l, 2))).astype(int)
                if s >= (T + 1): break
                if s + n_online_a >= (T + 1): n_online_a = T + 1 - s
                
                temp_array = sample_arm(a, n_online_a, theta, A)
                regret += n_online_a * (np.dot(A[optimal_arm] - A[a], theta))
                if j == 0 and not offline_samples_avail:
                    design_matrix = np.tile(A[a],[n_online_a, 1])
                    y_vec = temp_array
                else:
                    design_matrix = np.vstack((design_matrix, np.tile(A[a], [n_online_a, 1])))
                    y_vec = np.hstack((y_vec, temp_array))
                s += n_online_a
                j += 1

            hat_theta = np.linalg.lstsq(design_matrix, y_vec, rcond=None)[0]
            A_l = Eliminate(A_l, hat_theta, A, e_l)
            l += 1
        else:
            id = A_l[0]
            regret += np.dot(A[optimal_arm] - A[id], theta)
            s += 1
    return regret

def LinUCB_warm_start(d, A, theta, optimal_arm, non_zero_arm, T_o, V_pi_o, offline_data_dict, T):
    eps = .01
    s = 0
    l = 1
    inv_design_matrix = np.linalg.inv(l * np.identity(d) + T_o * V_pi_o)
    design_matrix = l * np.identity(d) + T_o * V_pi_o
    y_vec = np.zeros(d)
    for a in non_zero_arm:
        y_vec += np.sum(offline_data_dict[a]) * A[a]
        
    hat_theta = np.dot(inv_design_matrix, y_vec)
    log_det = np.trace(sp.linalg.logm(l * np.identity(d) + T_o * V_pi_o))
    det = np.linalg.det(design_matrix)
    m = np.sqrt(theta.T @ design_matrix @ theta)
    regret = 0
    
    while s < T + 1:
        beta = m + np.sqrt(2 * np.log(T) + np.log(np.linalg.det(design_matrix) / det)) 
        ucb = np.dot(A, hat_theta) + beta * np.linalg.norm(np.matmul(A, sp.linalg.sqrtm(inv_design_matrix)), axis=1)
        arm_id = np.argmax(ucb)
        t = log_det
        
        while t < log_det + np.log(1 + eps):
            sample = sample_arm(arm_id, 1, theta, A)
            y_vec += np.sum(sample) * A[arm_id]
            temp_vec = np.matmul(inv_design_matrix, A[arm_id])
            t += np.log(1 + np.dot(A[arm_id], temp_vec))
            design_matrix += np.outer(A[arm_id], A[arm_id])
            inv_design_matrix -= (1 / (1 + np.dot(A[arm_id], temp_vec))) * np.outer(temp_vec, temp_vec)
            s += 1
            regret += np.dot(A[optimal_arm] - A[arm_id], theta)
            if s > T: break
            
        log_det = t
        hat_theta = np.dot(inv_design_matrix, y_vec)
    return regret

def LinTS_warm_start(d, A, optimal_arm, theta, non_zero_arm, offline_data_dict, T_o, V_pi_o, T):
    s = 0
    eps = .01
    prior_cov = 1 * np.linalg.inv(np.identity(d) + T_o * V_pi_o)
    log_det = np.trace(sp.linalg.logm(np.identity(d) + T_o * V_pi_o))
    data_vec = np.zeros(d)
    rng = np.random.default_rng()
    v = np.sqrt(18 * d * np.log(T))
    
    for a in non_zero_arm:
        if a in offline_data_dict:
            data_vec += np.sum(offline_data_dict[a]) * A[a]
        
    prior_mean = np.matmul(prior_cov, data_vec)
    sample_theta = rng.multivariate_normal(prior_mean, np.power(v, 2) * prior_cov)
    regret = 0
    
    while s < T + 1:
        arm_id = np.argmax(np.matmul(A, sample_theta))
        t = log_det
        while t < log_det + np.log(1 + eps):
            sample = sample_arm(arm_id, 1, theta, A)
            data_vec += np.sum(sample) * A[arm_id]
            regret += np.dot(A[optimal_arm] - A[arm_id], theta)
            temp_vec = np.matmul(prior_cov, A[arm_id])
            prior_cov -= (1 / (1 + np.dot(A[arm_id], temp_vec))) * np.outer(temp_vec, temp_vec)
            prior_mean = np.matmul(prior_cov, data_vec)
            t += np.log(1 + np.dot(A[arm_id], temp_vec))
            s += 1
            if s > T: break
            
        sample_theta = rng.multivariate_normal(prior_mean, np.power(v, 2) * prior_cov)
        log_det = t
    return regret
