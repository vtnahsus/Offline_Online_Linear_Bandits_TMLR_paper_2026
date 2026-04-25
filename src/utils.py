import numpy as np
import math
from scipy.optimize import brentq

def find_optimal_arm(A,theta):
  optimal_arm_id=0
  optimal_arm_value=np.dot(A[optimal_arm_id],theta)
  for i in range(1,len(A)):
    t=np.dot(A[i],theta)
    if(t>optimal_arm_value):
      optimal_arm_id=i
      optimal_arm_value=t
  return optimal_arm_id


def suboptimality_gaparray(A,theta,optimal_arm):
  gap_array=np.array([])
  for i in range(len(A)):
    gap_i=np.dot(A[i],theta)-np.dot(A[optimal_arm],theta)
    gap_array=np.append(gap_array,[-gap_i])
  return gap_array

def find_suboptimality_gap(A,theta,optimal_arm):
  gap=float('-inf')
  for i in range(1,len(A)):
    if(i!=optimal_arm):
      gap_i=np.dot(A[i],theta)-np.dot(A[optimal_arm],theta)
      if(gap_i>gap):
        gap=gap_i
  return np.abs(gap)

# Problem generation function
# Input: d -dimension, K- No. of arms, mode="Uniform" or "EoO"
# mode parameter defines the arm generation criteria. "EoO" refers to End of Optimism example. "Uniform" refers to a uniform generation of arms. "Nonuniform" sets arms to be of different lengths.
# Output: theta-unknown parameter as numpy array, A-set of arms in a numpy matrix, optimal arm id as integer, suboptimality gap as float.
def problem_generation(d,K,mode):
  #generate theta
  theta= np.random.normal(size=(d,))
  theta=theta/np.linalg.norm(theta)
  if(mode=="Uniform"):
    A = np.random.normal(loc=0.0, scale=1.0, size=(K, d))
    A = A/np.linalg.norm(A,axis=1)[:,None]
  elif(mode=="EoO"):
    A=[]
    A.append(theta)
    v1=np.random.normal(size=(d,))
    v1=v1-np.dot(v1,theta)*theta
    v1=v1/np.linalg.norm(v1)
    v=theta+3e-1*v1
    v=v/np.linalg.norm(v)
    A.append(v)
    for i in range(K-2):
      v=np.random.normal(size=(d,))
      v=v-np.dot(v,theta)*theta
      v=v/np.linalg.norm(v)
      A.append(v)
    A=np.array(A)
  else:
    raise ValueError('mode parameter must be "Uniform" or "EoO"')
  optimal_arm=find_optimal_arm(A,theta)
  gap_subopt=find_suboptimality_gap(A,theta,optimal_arm)
  return(theta,A,optimal_arm,gap_subopt)

def sample_arm(arm_id,num_samples,theta,A):
  sample_vector=np.repeat(np.dot(A[arm_id],theta),num_samples)+np.random.normal(0,1,num_samples)
  return sample_vector

# Creates a random partition sample
def partition_sample(n,k):
  n_vec=np.arange(1,n)
  if(k==1):
    return np.array([n])
  subset = np.sort(np.random.choice(n_vec, size=k-1, replace=False))
  partition=[]
  for i in range(k):
    if(i==0):
      partition.append(subset[0])
    elif(i==k-1):
      partition.append(n-subset[i-1])
    else:
      partition.append(subset[i]-subset[i-1])
  return np.array(partition)

# Function that generates the offline data
# Input: T_o -total number of offline samples, non_zero_support-number of support points, theta, A-arm matrix, d- dimension.
# Output: A dictionary with keys as the arm indices and data as offline fraction, a dictionary with keys as arm indices with actual offline data, offline numpy gram matrix. non_zero_arm- a numpy array containing support points of offline data.

def offline_data_generation(T_o,non_zero_support,theta,A,d):
  if(non_zero_support>T_o):
    raise ValueError('non-zero offline support points exceed total offline samples')
  non_zero_arm=np.sort(np.random.choice(np.arange(len(A)), size=non_zero_support, replace=False))
  sample_vector=partition_sample(T_o,non_zero_support)
  offline_data_dict={}
  offline_frac=np.zeros(len(A))
  offline_frac_dict=dict.fromkeys(np.arange(len(A)),0)
  i=0
  for a in non_zero_arm:
    offline_data_dict[a]=sample_arm(a,sample_vector[i],theta,A)
    offline_frac_dict[a]=sample_vector[i]/T_o
    i=i+1

  V_pi_o=np.zeros(shape=(d,d))
  for a in non_zero_arm:
    V_pi_o=V_pi_o+offline_frac_dict[a]*np.outer(A[a],A[a])

  return(offline_frac_dict,offline_data_dict,V_pi_o,non_zero_arm)

# Function that regenerates the offline data when offline fraction exists.
# Input: T_o -total number of offline samples, non_zero_arms-number of support points, offline_frac_dict-fraction of offline data .
# Output: A dictionary with keys as arm indices with regenerated offline data.

def repeated_offline_data_generation(T_o,non_zero_arm,offline_frac_dict,theta,A):
  offline_data_dict={}
  offline_frac=np.zeros(len(A))
  temp_actual=0
  temp_floor=0
  m=len(non_zero_arm)
  i=0
  for a in non_zero_arm:
    sample_size=np.floor(offline_frac_dict[a]*T_o).astype(int)
    if(temp_actual-temp_floor>=1):
      sample_size=sample_size+1
    if(i==m-1):
      if(temp_floor+sample_size<T_o):
        slack=T_o-(temp_floor+sample_size)
        sample_size=sample_size+slack
    offline_data_dict[a]=sample_arm(a,sample_size,theta,A)
    temp_floor=temp_floor+sample_size
    temp_actual=temp_actual+offline_frac_dict[a]*T_o
    i=i+1
  return(offline_data_dict)

#compute d_effective
def compute_d_e(T_o,T,V_pi_o):
  eigenvalues=np.linalg.eigvals(V_pi_o)
  lambda_min=np.min(eigenvalues.real)
  if(T==0): return(0)
  if(T_o*lambda_min >0):
    return(np.minimum(np.sum(1/(1+(T_o/T)*eigenvalues.real)),T/(T_o*lambda_min)))
  else:
    return(np.sum(1/(1+(T_o/T)*eigenvalues.real)))

def find_toff(d_e,T,V_pi_o):
    objective = lambda x: compute_d_e(x,T,V_pi_o) - d_e
    root = brentq(objective, 0.001, T**3)
    return(math.ceil(root))
