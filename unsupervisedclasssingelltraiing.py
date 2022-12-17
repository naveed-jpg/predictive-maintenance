import numpy as np
np.set_printoptions(suppress=True, precision=4)
import pandas as pd
#from scipy.optimize import Bounds
#import test-train split
#from sklearn.model_selection import train_test_split
#import Logistic regression
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
from sklearn import preprocessing
#from sklearn.metrics import classification_report
#import seaborn as sns
import random
import matplotlib.pyplot as plt
from itertools import chain, combinations,product

# Load data here
H=pd.read_csv('/Users/inria/Desktop/M/TwinHouse.csv',sep=';',header=None)

# Inputs Temperatures
H=pd.read_excel('/Users/inria/Desktop/M/Experiment_1/Twin_house_exp1_house_O5_10min_ductwork_correction.xls',header=[0,1])
W=pd.read_csv('/Users/inria/Desktop/M/TwinWeather.csv',sep=';',header=None)
# Inputs
#T=H.iloc[:,1] #time
Ti2 = H.iloc[:,7];    # temperature in living room at 187 cm (output)
Ti1=H.iloc[:,6];      # temperature in living room at 125 cm
Ti=H.iloc[:,5];     #temperature in living room at 67cm
Tk = H.iloc[:,11];   # kitchen
Td = H.iloc[:,12];   # doorway
Tcr = H.iloc[:,8];    # corridor
Tchl = H.iloc[:,10];  # Children room
Tb=H.iloc[:,13];    #Bed room
Ta = H.iloc[:,3];    # attic
Tg = H.iloc[:,4];    # cellar
Tv = H.iloc[:,29];   # ventilation supply air 
To = W[2];    # outdoor
Qn = W[5];    # Solar radiations from north
Qs = W[7];    # specific global solar vert. South
Qw = W[8];    # specific global solar vert. West
Qi = H.iloc[:,20];
Qk=H.iloc[:,23]+H.iloc[:,24]; #Kithcne power input minus duct losses
Qd=H.iloc[:,25]; #Doorway Heater
QB=H.iloc[:,26]; #Bedroom Heater
df=pd.DataFrame([Ti,Ti1,Ti2,Tk,Td,Tcr,Tchl,Tb,Qi]).transpose()
df.columns=['Lvngrm67','Lvngrm125','Lvngrm187','Ktchn','Drwy','crrdr',
            'Chldrn_rm','Bed_rm','Power']
#Data Normalisation
df.iloc[:,0:8]=(df.iloc[:,0:8]-df.iloc[:,0:8].mean())/(df.iloc[:,0:8].max()-df.iloc[:,0:8].min())
a1=0.05
k=6
gam=0.9
df['Power1']=df['Power']>10
df['Power1']=df['Power1'].astype(int)

df.drop('Power',axis=1,inplace=True)
dfPower=df[df['Power1']==1]
dfNoPower=df[df['Power1']==0]
dfPower.drop('Power1',axis=1,inplace=True)
dfNoPower.drop('Power1',axis=1,inplace=True)
X_N=dfPower.sample(frac=0.5,replace=False)
X_M=dfPower.drop(X_N.index)
nn = NearestNeighbors(k).fit(X_N)
nn.fit(X_N,X_M)
dist, index = nn.kneighbors(X_M)
D=np.sum(np.power(dist,gam),1)
errorcount=0
anomaly=[]
for i in np.arange(0,len(dfNoPower),1):

    dist1,index1=nn.kneighbors(np.array(dfNoPower.iloc[i,0:8]).reshape(1,-1))
    D2=np.sum(np.power(dist1,gam),1)
    sumD=0
    for k in np.arange(0,len(X_M),1):
        if D2>D[k]:
            sumD=sumD+1
    check=sumD/len(X_M)
    check2=1-a1
    if sumD/len(X_M)>1-a1:
        anomaly.append(dfNoPower.index[i])
        error=1
        errorcount=errorcount+1
    else:
        error=0
        pvalue=1-check
av_error=errorcount/len(dfNoPower)


# Number of Sensors
s=np.arange(0,dfPower.shape[1],1) 
#Diiferent sensors combinations
x=list(chain(*(combinations(s,i) for i in range(1,1+len(s)))))
#Sampling rates
y=np.arange(dfPower.shape[0],1,-round(len(dfPower)/4))
# Numer of decimal places
z=np.arange(4,0,-1)
A=[x,y,z]
M=list(product(*A))

f_costbrt=[]
import time
start=time.time()
def f_cost(i):
    k=6
    #lmda=10 #wieght fot the # of sensors reeduction
    w1=10 #weight for the sensor energy
    lmda=100
    del1=[1,0.1,0.01,0.001,0.0001]##For digits after decial points
    rang=40-(-10) # Possible temp range
    n_bits=2*np.log2(rang/del1[M[i][2]]).round(0)# Number of bits req.for communication
    S_rate=M[i][1]/(3600*41*24) # sampling rate divided by number of days and hours per
    #day to calcualte hourly sampling rate
    E_batt=3.6e-6*S_rate*n_bits*len(M[i][0])*1e6 # Energy in microwatts
    
    dfPower.round(M[i][2])
    
    #random.seed(202)
    #random.sample(range(0,len(M)),5)
    #df1=df.sample(M[i][1])
    random.seed(101)
    df1Power=dfPower.iloc[random.sample(range(0,len(dfPower)),M[i][1]),:]
    #X_dat1=X_dat.iloc[random.sample(range(0,len(X)),M[i][1]),:]
    #X_train,X_test,y_train,y_test=train_test_split(df1.iloc[:,list(M[i][0])],
     #                           df1.TEMP,test_size=0.4,random_state=0)
    df1Power=df1Power.iloc[:,list(M[i][0])]
    X_dat1=dfNoPower.iloc[:,list(M[i][0])]
    N=round(len(df1Power)/2)
    F=len(df1Power)-N
    X_N=df1Power.iloc[0:N,:]
    X_M=df1Power.iloc[N:N+F,:]
    nn = NearestNeighbors(k).fit(X_N)
    nn.fit(X_N,X_M)
    dist, index = nn.kneighbors(X_M)
    D=np.sum(np.power(dist,gam),1)
    errorcount=0
    anomaly1=[]
    for i1 in np.arange(0,len(X_dat1),1):
        
        dist1,index1=nn.kneighbors(np.array(X_dat1.iloc[i1,:]).reshape(1,-1))
        D2=np.sum(np.power(dist1,gam),1)
        sumD=0
        for k in np.arange(0,F,1):
            if D2>D[k]:
                sumD=sumD+1
        check=sumD/F
        check2=1-a1
        if sumD/F>1-a1:
            anomaly1.append(X_dat1.index[i1])
            error=1
            errorcount=errorcount+1
        else:
            error=0
            pvalue=1-check
    av_error1=errorcount/len(X_dat1)
    accuracy=len(np.intersect1d(anomaly,anomaly1))/len(anomaly)
    return -w1*E_batt+lmda*accuracy,accuracy,E_batt,anomaly1
    #print(f_cost)
#Cahnge step size here:

epsi=10
epsi

#Initiate an alpha
alpha=np.repeat(0.5,len(M))
g=np.zeros(len(alpha))
random.seed()
#o=random.sample(range(0,len(M)),1)
#lmda=np.arange(10,120,10)
for a in range(1):
    alpha=np.repeat(0.5,len(M))
    g=np.repeat(0.5,len(M))
    o=random.sample(range(0,len(M)),50)
    alpha1=[]
    Max1=[]
    Max2=[]
    Max3=[]
    Max4=[]
    M12=[]
    anomaly2=[]
    for i in o:
        deg=epsi*f_cost(i)[0]
        for x in range(0,len(alpha)):
            if x==i:
                g[x]=(np.sum(alpha)-alpha[x])/(np.sum(alpha)**2)
            else:
                g[x]=-alpha[i]/(np.sum(alpha)**2)       
        alpha=alpha+deg*g
        Max1.append(np.argmax(alpha))
        Max2.append(f_cost(np.argmax(alpha))[1])
        Max3.append(f_cost(np.argmax(alpha))[0])
        Max4.append(f_cost(np.argmax(alpha))[2])
        M12.append(M[np.argmax(alpha)])
        anomaly2.append(f_cost(np.argmax(alpha))[3])
        alpha1.append(alpha)
    end=time.time()
    tot=end-start
    tot=np.repeat(tot,len(Max2))
    #np.savetxt('accuracySDG10epsi.txt',Max2)
    #np.savetxt('argmaxSDG10epsi.txt',Max1)
    #np.savetxt('execTimSDG10epsi.txt',tot)
    rs=pd.DataFrame(np.array([Max1,Max2,Max3,Max4,M12,tot]).transpose(),columns=['argmax','accuracy','communication cost','Battery Energy','communication policy','time'])
    rs.to_csv('/Users/inria/Desktop/Predictive Maintenance/constep1/K20SemiSupervised%a'%a,index=False)

fig,ax1=plt.subplots()
ax1.plot(df.iloc[:,2:5],label=['T1','T2','T3'])
ax1.legend(loc='upper right')

ax2=ax1.twinx()
ax2.plot(df['Power'],label='Input Power',color='red')
ax2.legend(loc='lower right')
ax1.set_xlabel('No. of Iterations')
ax1.set_ylabel('Temperature [C]')
#ax1.yaxis.label.set_color('red')
#ax1.tick_params(axis='y', colors='red')

ax2.tick_params(axis='y',colors='red')
ax2.spines['right'].set_color('red')
#ax2.spines['left'].set_color('red')

#ax2.yaxis.label.set_color('green')
#ax2.set_ylabel('Battery Power [\u03bcwatts]')
ax2.set_ylabel('Input Power[Watts]')
plt.show()
#ax.legend('Battery Power Curve')