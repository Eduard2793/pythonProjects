import numpy as np

def sigmoid(x):
    return 1/(1 + np.exp(-1*x))

def sigmoid_prime(x):
    return (sigmoid(x)*(1 - sigmoid(x)))

def cross_entropy(Y,Y_hat): # np.array (m,1)
    q = Y*np.log(Y_hat) + (1 - Y)*np.log(1 - Y_hat)
    return(-q)
    

def cross_entropy_derivative(Y,Y_hat): # grad by a
    return (((1-Y)*Y_hat - (1-Y_hat)*Y)/(Y_hat*(1-Y_hat)))
    

def quadratic(Y,Y_hat):
    q = Y-Y_hat
    return ((q**2)/2)

def quadratic_derivative(Y,Y_hat):
    return Y-Y_hat
    
class Neuron:
    
    def __init__(self,w,b,activeF):
        self.w = w # shape = (m,1)
        self.b = b
        self.activeF = activeF
        
    def summatoryF(self,X): # shape = (m,1)
        return self.w.T.dot(X) + self.b #==========

    def activation(self,X):
        return self.activeF(self.summatoryF(X))
        
    def __str__(self):
        return self.w.__str__() + "  " + self.b.__str__()

class Network:
    
    def __init__(self,n_array,costF,costF_derivative,activeF_derivative):
        #n_array contains neuron num in each layer
        self.m_of_n = []
        self.costF = costF
        self.gradJ_a = costF_derivative
        self.activeF_derivative = activeF_derivative
        for i in range(0,len(n_array)):
            L = []
            for j in range(0,n_array[i]):
                if i == 0:
                    L.append(0)
                else:
                    m = n_array[i-1]
                    w = 2 * np.random.random((m, 1)) - 1#np.zeros((m,1))#
                    b = 0#np.random.random()
                    L.append(Neuron(w,b,activeF = sigmoid))
            self.m_of_n.append(L)
        

    
    def forward_propogation(self,X):
        a = []
        n = len(self.m_of_n)
        for i in range(0,n):
            a_singl = []
            for j in range(0,len(self.m_of_n[i])):
                if i == 0:
                    self.m_of_n[i][j] = X[j]
                    a_singl.append(X[j])
                else:
                    xl = np.array(a[i-1]).reshape(len(a[i-1]),1)
                    a_singl.append(self.m_of_n[i][j].activation(xl))
            a.append(a_singl)
        self.a = a
    
    
    def z_L(self,L):
        z = []        
        xl = np.array(self.a[L-1]).reshape(len(self.a[L-1]),1)
        for i in range(0,len(self.m_of_n[L])):
            z.append(self.m_of_n[L][i].summatoryF(xl))
        return np.array(z).reshape(len(z),1)       

    def get_Last_delta(self,Y):
        a_Last = self.a[len(self.a)-1]
        a_Last = np.array(a_Last).reshape(len(a_Last),1)
        z_Last = self.z_L(len(self.m_of_n)-1)
        return self.gradJ_a(Y,a_Last) * self.activeF_derivative(z_Last)

    def get_delta_L(self,delta_front, sum_this, w_front):
        return w_front.T.dot(delta_front) * self.activF_derivative(sum_this)

    def backword(self,X,Y):
        n_L = len(self.m_of_n)
        self.forward_propogation(X)
        deltas = []
        for i in range(n_L-1,-1,-1):
            if i == n_L-1:
                deltas.append(self.get_Last_delta(Y))
            elif i == 0:
                deltas.append(np.zeros((len(self.m_of_n[i]),1))) #?
            else:
                d = deltas[n_L-2-i]
                u = len(self.m_of_n[i+1])#i
                y = len(self.m_of_n[i]) #i-1
                w_front = np.zeros((u,y))
                for k in range(0,u):
                    for j in range(0,y):
                        w_front[k][j] = self.m_of_n[i][k].w[j].copy()
                deltas.append(self.get_delta_L(d,self.z_L(i),w_front))
                
        self.errors_on_single_example = []
        for L in range(len(deltas)-1,-1,-1):
            lay = []
            for j in range(0,len(deltas[L])):
                lay.append(deltas[L][j])
            self.errors_on_single_example.append(lay)

    def train_on_single_example(self,X,Y,learning_rate):
        self.backword(X,Y)
        deltas = self.errors_on_single_example
        for L in range(1,len(deltas)):
            for j in range(0,len(deltas[L])):
                self.m_of_n[L][j].w -= self.a[L-1]*deltas[L][j]*learning_rate
                #self.m_of_n[L][j].b -= deltas[L][j]*learning_rate
                
    def weightPrint(self):
        s = []
        for i in range(0,len(self.m_of_n)):
            l = []
            for j in range(0,len(self.m_of_n[i])):
                l.append(self.m_of_n[i][j].__str__())
            s.append(l)
        return s.__str__()
        

    def activePrint(self):
        s = []
        for i in range(0,len(self.a)):
            l = []
            for j in range(0,len(self.a[i])):
                l.append(self.a[i][j].__str__())
            s.append(l)
        return s.__str__()
'''
N = Network([2,1],costF = cross_entropy,costF_derivative = cross_entropy_derivative,activeF_derivative = sigmoid_prime)
X1 = np.array([1,0.3]).reshape(2,1)
Y1 = np.array([[1]])
X2 = np.array([0.4,0.5]).reshape(2,1)
Y2 = np.array([[1]])
X3 = np.array([0.7,0.8]).reshape(2,1)
Y3 = np.array([[0]])
N.train_on_single_example(X1,1,1)
N.train_on_single_example(X2,1,1)
N.train_on_single_example(X3,0,1)
print(N.weightPrint())
print()
print(N.activePrint())
print()
'''
n = 10
m = 5
X = 20 * np.random.sample((n, m)) - 10
y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]

N1 = Network([m,1],costF = quadratic,costF_derivative = quadratic_derivative,activeF_derivative = sigmoid_prime)

for i in range(0,n):
    N1.train_on_single_example(X[i].reshape(m,1),y[i],0.1)
'''
X1 = np.array([1,0.3]).reshape(2,1)
Y1 = np.array([[1]])
X2 = np.array([0.4,0.5]).reshape(2,1)
Y2 = np.array([[1]])
X3 = np.array([0.7,0.8]).reshape(2,1)
Y3 = np.array([[0]])
N1.train_on_single_example(X1,1,1)
N1.train_on_single_example(X2,1,1)
N1.train_on_single_example(X3,0,1)
'''
print(N1.weightPrint())
print()
print(N1.activePrint())

