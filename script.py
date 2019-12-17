"""
Miniproject - Advanced Probibalistic Machine Learning

"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.stats import norm
import time

print("Q4.1")

M = np.array([[1, -1]])
sigma1, sigma2, sigma3 = 25/3, 25/3, 25/6
mu1, mu2 = 25, 25
mu = np.array([[mu1], [mu2]])
sigmaA = np.array([[sigma1**2, 0], [0, sigma2**2]])
sigmaAB = np.linalg.inv(np.linalg.inv(sigmaA)+np.transpose(M)*(1/sigma3**2)@M)

myclip_a = 0
myclip_b = 1000
my_mean = mu1 - mu2
my_std = sigma3
a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

# Gibbs sampling
K = 1000
s1 = np.zeros(K)
s2 = np.zeros(K)
t = np.zeros(K)
s1[0] = 1
s2[0] = 1
t[0] = 0
for k in range(K-1):
  muAB = sigmaAB@(np.linalg.inv(sigmaA)@mu+np.transpose(M)*(t[k]/sigma3**2))
  muAB2 = np.ravel(muAB)
  s1[k+1], s2[k+1] = np.random.multivariate_normal(muAB2, sigmaAB)
  my_mean = s1[k+1] - s2[k+1]
  a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
  t[k+1] = truncnorm.rvs(a, b, my_mean, my_std)
  

plt.plot(s1,label="s1")
plt.plot(s2,label="s2")
plt.legend()
plt.show()

#***************************************************************************

print("Q4.2")
  
#Parameters
x = np.linspace(mu1-5*sigma1,mu1+5*sigma1,100)
mu_s1 = np.mean(s1)
std_s1 = np.std(s1)
S1 = norm.pdf(x, mu_s1, std_s1)

mu_s2 = np.mean(s2)
std_s2 = np.std(s2)
S2 = norm.pdf(x, mu_s2, std_s2)

#Make a plot
plt.plot(x, S1, linewidth=2, label="s1")
plt.plot(x, S2, linewidth=2, label="s2")
plt.legend()
plt.show()

#***************************************************************************

print("Q4.3")

time_start = time.clock()
#run your code
time_elapsed = (time.clock() - time_start)

# Gibbs sampling
K = 1000
s1 = np.zeros(K)
s2 = np.zeros(K)
t = np.zeros(K)
s1[0] = 25
s2[0] = 25
t[0] = 0
for k in range(K-1):
  muAB = sigmaAB@(np.linalg.inv(sigmaA)@mu+np.transpose(M)*(t[k]/sigma3**2))
  muAB2 = np.ravel(muAB)
  s1[k+1], s2[k+1] = np.random.multivariate_normal(muAB2, sigmaAB)
  my_mean = s1[k+1] - s2[k+1]
  a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
  t[k+1] = truncnorm.rvs(a, b, my_mean, my_std)
  
time_elapsed = (time.clock() - time_start)
print(time_elapsed, "seconds to run with", K, "samples")

#Parameters
x = np.linspace(mu1-5*sigma1,mu1+5*sigma1,100)
mu_s1 = np.mean(s1)
std_s1 = np.std(s1)
S1 = norm.pdf(x, mu_s1, std_s1)

mu_s2 = np.mean(s2)
std_s2 = np.std(s2)
S2 = norm.pdf(x, mu_s2, std_s2)

#Make a plot
plt.hist(s1, label="s1", bins =50, density=True)
plt.plot(x, S1, linewidth=2, label="Approx. Gaussian")
plt.legend()
plt.show()
plt.hist(s2, label="s2", bins =50, density=True)
plt.plot(x, S2, linewidth=2, label="Approx. Gaussian")
plt.legend()
plt.show()

#***************************************************************************

print("Q4.4")

# Gibbs sampling
K = 1000
s1 = np.zeros(K)
s2 = np.zeros(K)
t = np.zeros(K)
s1[0] = 1
s2[0] = 1
t[0] = 0
for k in range(K-1):
  muAB = sigmaAB@(np.linalg.inv(sigmaA)@mu+np.transpose(M)*(t[k]/sigma3**2))
  muAB2 = np.ravel(muAB)
  s1[k+1], s2[k+1] = np.random.multivariate_normal(muAB2, sigmaAB)
  my_mean = s1[k+1] - s2[k+1]
  a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
  t[k+1] = truncnorm.rvs(a, b, my_mean, my_std)
  
x = np.linspace(mu1-5*sigma1,mu1+5*sigma1,100)

#Parameters to set
mu_s1 = np.mean(s1)
std_s1 = np.std(s1)
S1 = norm.pdf(x, mu_s1, std_s1)
S1_start = norm.pdf(x, mu1, sigma1)

mu_s2 = np.mean(s2)
std_s2 = np.std(s2)
S2 = norm.pdf(x, mu_s2, std_s2)
S2_start = norm.pdf(x, mu2, sigma2)

#Make a plot
plt.plot(x,S1, linewidth=2, label = "posterior")
plt.plot(x,S1_start, linewidth = 2, label = "prior")
plt.title('S1')
plt.legend()
plt.show() 

plt.plot(x,S2, linewidth=2,label = "posterior")
plt.plot(x,S2_start, linewidth = 2,label = "prior")
plt.title('S2')
plt.legend()
plt.show()

#***************************************************************************

print("Q5")

def GibbsSampling(mu1,sigma1,mu2,sigma2):
    M = np.array([[1, -1]])
    sigma3 = (25/6)

    mu = np.array([[mu1], [mu2]])
    sigmaA = np.array([[sigma1**2, 0], [0, sigma2**2]])
    sigmaAB = np.linalg.inv(np.linalg.inv(sigmaA)+np.transpose(M)*(1/sigma3**2)@M)

    myclip_a = 0
    myclip_b = 1000
    my_mean = mu1 - mu2
    my_std = sigma3
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    # Gibbs sampling
    K = 1000
    s1 = np.zeros(K)
    s2 = np.zeros(K)
    t = np.zeros(K)
    s1[0] = 1
    s2[0] = 1
    t[0] = 0
    for k in range(K-1):
      muAB = sigmaAB@(np.linalg.inv(sigmaA)@mu+np.transpose(M)*(t[k]/sigma3**2))
      muAB2 = np.ravel(muAB)
      s1[k+1], s2[k+1] = np.random.multivariate_normal(muAB2, sigmaAB)
      my_mean = s1[k+1] - s2[k+1]
      a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
      t[k+1] = truncnorm.rvs(a, b, my_mean, my_std)
    
    #Parameters
    mu_s1 = np.mean(s1)
    std_s1 = np.std(s1)

    mu_s2 = np.mean(s2)
    std_s2 = np.std(s2)

    return(mu_s1, std_s1, mu_s2, std_s2)

        
data = pd.read_csv("SerieA.csv") 
teams = data['team1'].drop_duplicates()
data = data.sample(frac=1).reset_index(drop=True) #gives random data. comment out to have sorted csv-file
team_list = dict()

for team in teams:
    team_list[team] = (25,(25/3)) #default (mean,variance) for each team
    
for i in range(data.shape[0]):
    score = data.iloc[i]['score1'] - data.iloc[i]['score2']
    if score > 0:
        winner = data.iloc[i]['team1']
        looser = data.iloc[i]['team2']
        mu1, sigma1, mu2, sigma2 = GibbsSampling(team_list[winner][0], team_list[winner][1],team_list[looser][0], team_list[looser][1])
        team_list[winner] = [mu1, sigma1]
        team_list[looser] = [mu2, sigma2]
    
    elif score < 0:
        winner = data.iloc[i]['team2']
        looser = data.iloc[i]['team1']
        mu1, sigma1, mu2, sigma2 = GibbsSampling(team_list[winner][0], team_list[winner][1],team_list[looser][0], team_list[looser][1])
        team_list[winner] = [mu1, sigma1]
        team_list[looser] = [mu2, sigma2]

df = pd.DataFrame.from_dict(team_list, orient='index', columns=['Skill', 'Standard Deviation'])
df.sort_values(by='Skill', inplace=True, ascending = False)
df['Rank'] = range(1, len(df) + 1)
print(team_list)

#***************************************************************************

print("Q6")

def GibbsSampling(mu1,sigma1,mu2,sigma2):
    M = np.array([[1, -1]])
    sigma3 = (25/6)

    mu = np.array([[mu1], [mu2]])
    sigmaA = np.array([[sigma1**2, 0], [0, sigma2**2]])
    sigmaAB = np.linalg.inv(np.linalg.inv(sigmaA)+np.transpose(M)*(1/sigma3**2)@M)

    myclip_a = 0
    myclip_b = 1000
    my_mean = mu1 - mu2
    my_std = sigma3
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    # Gibbs sampling
    K = 1000
    s1 = np.zeros(K)
    s2 = np.zeros(K)
    t = np.zeros(K)
    s1[0] = 1
    s2[0] = 1
    t[0] = 0
    for k in range(K-1):
      muAB = sigmaAB@(np.linalg.inv(sigmaA)@mu+np.transpose(M)*(t[k]/sigma3**2))
      muAB2 = np.ravel(muAB)
      s1[k+1], s2[k+1] = np.random.multivariate_normal(muAB2, sigmaAB)
      my_mean = s1[k+1] - s2[k+1]
      a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
      t[k+1] = truncnorm.rvs(a, b, my_mean, my_std)
    
    #Parameters
    mu_s1 = np.mean(s1)
    std_s1 = np.std(s1)

    mu_s2 = np.mean(s2)
    std_s2 = np.std(s2)

    return(mu_s1, std_s1, mu_s2, std_s2)


def predict(mu1, sigma1, mu2, sigma2):
    L = 1000 #number of samples

    s1 = np.random.normal(mu1, sigma1, L)
    s2 = np.random.normal(mu2, sigma2, L)
    
    t = np.mean(s1 - s2)
    y = np.sign(t)
    result_list.append(y)
    
    return(result_list)
    
    
data = pd.read_csv("SerieA.csv") 
teams = data['team1'].drop_duplicates()
team_list = dict()
result_list = list()

for team in teams:
    team_list[team] = (25,(25/3))
    
for i in range(data.shape[0]):
    score = data.iloc[i]['score1'] - data.iloc[i]['score2']
    predict(team_list[data.iloc[i]['team1']][0], team_list[data.iloc[i]['team1']][1],team_list[data.iloc[i]['team2']][0], team_list[data.iloc[i]['team2']][1])
    if score > 0:
        winner = data.iloc[i]['team1']
        looser = data.iloc[i]['team2']
        mu1, sigma1, mu2, sigma2 = GibbsSampling(team_list[winner][0], team_list[winner][1],team_list[looser][0], team_list[looser][1])
        team_list[winner] = [mu1, sigma1]
        team_list[looser] = [mu2, sigma2]
    
    elif score < 0:
        winner = data.iloc[i]['team2']
        looser = data.iloc[i]['team1']
        mu1, sigma1, mu2, sigma2 = GibbsSampling(team_list[winner][0], team_list[winner][1],team_list[looser][0], team_list[looser][1])
        team_list[winner] = [mu1, sigma1]
        team_list[looser] = [mu2, sigma2]


def compare_result(prediction, data):
    true_result = list()
    count = 0
    score = data['score1'] - data['score2']
    for i in range(data.shape[0]):
        if score[i] > 0:
            true_result.append(1)
        elif score[i] < 0:
            true_result.append(-1)
        else:
            true_result.append(0)
    
    for k in range(len(true_result)):
        if true_result[k] == prediction[k]:
            count += 1

    draw = true_result.count(0)
    pred_rate = count/380
    pred_rate_discard_zeros = count/(380-draw)
    print("Number of correct predictions:", count)
    print("Prediction rate:", pred_rate)
    print("Prediction rate if disregarding draws:", pred_rate_discard_zeros)
    return()
    
compare_result(result_list, data) 

#***************************************************************************

print("Q8")

def multiplyGauss(m1,s1,m2,s2):
    s = 1/(1/s1+1/s2)
    m = (m1/s1+m2/s2)*s
    return m, s

def divideGauss(m1,s1,m2,s2):
    m,s = multiplyGauss(m1,s1,m2,-s2)
    return m, s

def truncGaussMM(my_a,my_b,m1,s1):
    a, b = (my_a - m1) / np.sqrt(s1), (my_b - m1) / np.sqrt(s1)
    m = truncnorm.mean(a, b, loc=m1, scale=np.sqrt(s1))
    s = truncnorm.var(a, b, loc=0, scale=np.sqrt(s1))
    return m, s

# Defining the 5 hyperparameters
# y0 = 1 indicates that player 1 won a game
m1 = 25
m2 = 25
v1 = (25/3)**2
v2 = (25/3)**2
v3 = (25/6)**2
y0 = 1

#--------STARTING THE MESSAGE PASSING FROM THE TOP-----

# Message from factor f_s1 to node s1
mu_fs1_s1_m = m1
mu_fs1_s1_v = v1

# Message from factor f_s2 to node s2
mu_fs2_s2_m = m2
mu_fs2_s2_v = v2

# Message from node s1 to factor f_s1s2w
mu_s1_s1s2w_m = mu_fs1_s1_m
mu_s1_s1s2w_v = mu_fs1_s1_v

# Message from node s2 to factor f_s1s2w
mu_s2_fs1s2w_m = mu_fs2_s2_m
mu_s2_fs1s2w_v = mu_fs2_s2_v

# Message from factor f_s1s2w to node w
mu_fs1s2w_w_m = mu_fs1_s1_m - mu_fs2_s2_m
mu_fs1s2w_w_v = mu_fs1_s1_v + mu_fs2_s2_v

# Message from node w to factor f_wt
mu_w_fwt_m = mu_fs1s2w_w_m
mu_w_fwt_v = mu_fs1s2w_w_v

# Message from factor f_wt to node t
mu_fwt_t_m = mu_w_fwt_m
mu_fwt_t_v = mu_w_fwt_v + v3

# Do moment matching of the marginal of t
if y0==1:
    a, b = 0, 1000
else:
    a, b = -1000, 0

#Turning the truncated Gaussian into a Gaussian
pt_m, pt_v = truncGaussMM(a,b,mu_fwt_t_m,mu_fwt_t_v)

#Compute the updated message from f_yt to t
mu_fyt_t_m, mu_fyt_t_v = divideGauss(pt_m,pt_v,mu_fwt_t_m,mu_fwt_t_v)

#-------SENDING MESSAGES BACK UPWARDS------------
# Compute the message from node t to factor f_wt
mu_t_fwt_m = mu_fyt_t_m
mu_t_fwt_v = mu_fyt_t_v

# Compute the message from factor f_wt to node w
mu_fwt_x_m = mu_t_fwt_m
mu_fwt_x_v = mu_t_fwt_v + v3

# Compute the message from node w to factor f_s1s2w
mu_w_fs1s2w_m = mu_fwt_x_m
mu_w_fs1s2w_v = mu_fwt_x_v

# Compute the message from factor f_s1s2w to node s1
mu_fs1s2w_s1_m = mu_w_fs1s2w_m - m2
mu_fs1s2w_s1_v = mu_w_fs1s2w_v + v2

# Compute the message from factor f_s1s2w to node s2
mu_fs1s2w_s2_m = mu_w_fs1s2w_m - m1
mu_fs1s2w_s2_v = mu_w_fs1s2w_v + v1

# Compute the marginal of s1 and s2
ps1_m, ps1_v = multiplyGauss(mu_fs1_s1_m,mu_fs1_s1_v,mu_fs1s2w_s1_m,mu_fs1s2w_s1_v)
ps2_m, ps2_v = multiplyGauss(mu_fs2_s2_m,mu_fs2_s2_v,mu_fs1s2w_s2_m,mu_fs1s2w_s2_v)

#--------MAKING THE PLOT------
#Parameters
L = 100 #number of samples
x = np.linspace(m1-v1, m1+v1, 100)

#Draw values from the Gaussian distributions
s1_norm = np.random.normal(ps1_m, np.sqrt(ps1_v), L)
s2_norm = np.random.normal(ps2_m, np.sqrt(ps2_v), L)

mu_s1 = np.mean(s1_norm)
mu_s2 = np.mean(s2_norm)
var_s1 = np.var(s1_norm)
var_s2 = np.var(s2_norm)

#Making pdf:s
s1_pdf = norm.pdf(x, mu_s1, np.sqrt(var_s1))
s2_pdf = norm.pdf(x, mu_s2, np.sqrt(var_s2))

#Make a plot
plt.hist(s1, label="s1, Gibbs sampling", bins =50, density=True)
plt.plot(x, s1_pdf, linewidth=2, label="s1, message passing")
plt.legend()
plt.show()
plt.hist(s2, label="s2, Gibbs sampling", bins =50, density=True)
plt.plot(x, s2_pdf, linewidth=2, label="s2, message passing")
plt.legend()
plt.show()

#***************************************************************************

print("Q9")

def GibbsSampling(mu1,sigma1,mu2,sigma2):
    M = np.array([[1, -1]])
    sigma3 = (25/6)

    mu = np.array([[mu1], [mu2]])
    sigmaA = np.array([[sigma1**2, 0], [0, sigma2**2]])
    sigmaAB = np.linalg.inv(np.linalg.inv(sigmaA)+np.transpose(M)*(1/sigma3**2)@M)

    myclip_a = 0
    myclip_b = 1000
    my_mean = mu1 - mu2
    my_std = sigma3
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    # Gibbs sampling
    K = 1000
    s1 = np.zeros(K)
    s2 = np.zeros(K)
    t = np.zeros(K)
    s1[0] = 1
    s2[0] = 1
    t[0] = 0
    for k in range(K-1):
      muAB = sigmaAB@(np.linalg.inv(sigmaA)@mu+np.transpose(M)*(t[k]/sigma3**2))
      muAB2 = np.ravel(muAB)
      s1[k+1], s2[k+1] = np.random.multivariate_normal(muAB2, sigmaAB)
      my_mean = s1[k+1] - s2[k+1]
      a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
      t[k+1] = truncnorm.rvs(a, b, my_mean, my_std)
    
    #Parameters
    s1 = np.delete(s1, 2, 0) #remove burn-in
    s2 = np.delete(s2, 2, 0) #remove burn-in
    mu_s1 = np.mean(s1)
    std_s1 = np.std(s1)

    mu_s2 = np.mean(s2)
    std_s2 = np.std(s2)

    return(mu_s1, std_s1, mu_s2, std_s2)

        
data = pd.read_csv("NBA1314_Games.csv") 
teams = data['team1'].drop_duplicates()
data = data.sample(frac=1).reset_index(drop=True) #gives random data. comment out to have sorted csv-file
team_list = dict()

for team in teams:
    team_list[team] = (25,(25/3)) #default (mean,variance) for each team
    
for i in range(data.shape[0]):
    score = data.iloc[i]['score1'] - data.iloc[i]['score2']
    if score > 0:
        winner = data.iloc[i]['team1']
        #print(winner)
        looser = data.iloc[i]['team2']
        mu1, sigma1, mu2, sigma2 = GibbsSampling(team_list[winner][0], team_list[winner][1],team_list[looser][0], team_list[looser][1])
        team_list[winner] = [mu1, sigma1]
        team_list[looser] = [mu2, sigma2]
    
    elif score < 0:
        winner = data.iloc[i]['team2']
        #print(winner)
        looser = data.iloc[i]['team1']
        mu1, sigma1, mu2, sigma2 = GibbsSampling(team_list[winner][0], team_list[winner][1],team_list[looser][0], team_list[looser][1])
        team_list[winner] = [mu1, sigma1]
        team_list[looser] = [mu2, sigma2]


df = pd.DataFrame.from_dict(team_list, orient='index', columns=['Skill', 'Standard Deviation'])
df.sort_values(by='Skill', inplace=True, ascending = False)
df['Rank'] = range(1, len(df) + 1)
print(team_list)

#***************************************************************************

print("Q10", "\n")


def GibbsSampling(mu1,sigma1,mu2,sigma2):
    M = np.array([[1, -1]])
    sigma3 = (25/6)

    mu = np.array([[mu1], [mu2]])
    sigmaA = np.array([[sigma1**2, 0], [0, sigma2**2]])
    sigmaAB = np.linalg.inv(np.linalg.inv(sigmaA)+np.transpose(M)*(1/sigma3**2)@M)

    myclip_a = 0
    myclip_b = 1000
    my_mean = mu1 - mu2
    my_std = sigma3
    a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std

    # Gibbs sampling
    K = 1000
    s1 = np.zeros(K)
    s2 = np.zeros(K)
    t = np.zeros(K)
    s1[0] = 1
    s2[0] = 1
    t[0] = 0
    for k in range(K-1):
      muAB = sigmaAB@(np.linalg.inv(sigmaA)@mu+np.transpose(M)*(t[k]/sigma3**2))
      muAB2 = np.ravel(muAB)
      s1[k+1], s2[k+1] = np.random.multivariate_normal(muAB2, sigmaAB)
      my_mean = s1[k+1] - s2[k+1]
      a, b = (myclip_a - my_mean) / my_std, (myclip_b - my_mean) / my_std
      t[k+1] = truncnorm.rvs(a, b, my_mean, my_std)
    
    #Parameters
    s1 = np.delete(s1, 2, 0) #remove burn-in
    s2 = np.delete(s2, 2, 0) #remove burn-in
    mu_s1 = np.mean(s1)
    std_s1 = np.std(s1)

    mu_s2 = np.mean(s2)
    std_s2 = np.std(s2)

    return(mu_s1, std_s1, mu_s2, std_s2)


def predict(mu1, sigma1, mu2, sigma2, i):
    L = 1000 #number of samples
    s1 = np.random.normal(mu1, sigma1, L)
    s2 = np.random.normal(mu2, sigma2, L)
    t = np.mean(s1 - s2)
    if i > 19 and abs(t) < 1: 
        y = 0
        result_list.append(y)
    else:
        y = np.sign(t)
        result_list.append(y)
    
    return(result_list)
    
    
data = pd.read_csv("SerieA.csv") 
teams = data['team1'].drop_duplicates()
team_list = dict()
result_list = list()


for team in teams:
    team_list[team] = (25,(25/3))
    
for i in range(data.shape[0]):
    score = data.iloc[i]['score1'] - data.iloc[i]['score2']
    predict(team_list[data.iloc[i]['team1']][0], team_list[data.iloc[i]['team1']][1],team_list[data.iloc[i]['team2']][0], team_list[data.iloc[i]['team2']][1], i)
    if score > 0:
        winner = data.iloc[i]['team1']
        looser = data.iloc[i]['team2']
        mu1, sigma1, mu2, sigma2 = GibbsSampling(team_list[winner][0], team_list[winner][1],team_list[looser][0], team_list[looser][1])
        team_list[winner] = [mu1, sigma1]
        team_list[looser] = [mu2, sigma2]
    
    elif score < 0:
        winner = data.iloc[i]['team2']
        looser = data.iloc[i]['team1']
        mu1, sigma1, mu2, sigma2 = GibbsSampling(team_list[winner][0], team_list[winner][1],team_list[looser][0], team_list[looser][1])
        team_list[winner] = [mu1, sigma1]
        team_list[looser] = [mu2, sigma2]


def compare_result(prediction, data):
    true_result = list()
    prediction_result = np.around(prediction)
    count = 0
    score = data['score1'] - data['score2']
    for i in range(data.shape[0]):
        if score[i] > 0:
            true_result.append(1)
        elif score[i] < 0:
            true_result.append(-1)
        else:
            true_result.append(0)
    
    for k in range(len(true_result)):
        if true_result[k] == prediction_result[k]:
            count += 1
    draw = true_result.count(0)
    pred_rate = count/380
    draw = true_result.count(0)
    pred_rate = count/380
    pred_rate_discard_zeros = count/(380-draw)
    print("Number of correct predictions:", count)
    print("Prediction rate:", pred_rate)
    print("Prediction rate if disregarding draws:", pred_rate_discard_zeros)

    return()
    
compare_result(result_list, data)
    