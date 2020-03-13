import numpy as np

def sigmoid(x):    
    return 1/(1 + np.exp(-10*x))

def sigmoid_prime(x):
    return (10*sigmoid(x)*(1 - sigmoid(x)))

def cross_entropy(Y,Y_hat): # np.array (m,1)
    q = Y*np.log(Y_hat) + (1 - Y)*np.log(1 - Y_hat)
    return(-q)
    

def cross_entropy_deriv(Y,Y_hat): # grad by a
    return (((1-Y)*Y_hat - (1-Y_hat)*Y)/(Y_hat*(1-Y_hat)))
    

def quadratic(Y,Y_hat):
    q = Y-Y_hat
    return ((q**2)/2)

def quadratic_deriv(Y,Y_hat):
    return Y - Y_hat

class Neuron:

    def __init__(self,w,b,activ):
        self.w = w
        self.b = b
        self.activF = activ

    def summatory(self,x):
        s = 0
        for i in range(0,len(x)):
            s += self.w[i]*x[i]
        z = s + self.b
        return z

    def activate(self,x):
        return self.activF(self.summatory(x))

    def __str__(self):
        return self.w.__str__() + "  " + self.b.__str__()
        
class Discriminator:

    def __init__(self,net_size,activ,activ_deriv,cost,cost_deriv):
        self.activ_deriv = activ_deriv
        #self.cost = cost # в принципе не для валидации не нужна
        self.cost_deriv = cost_deriv
        self.N = [] # matrix of neurons
        self.A = [] # matrix of activations
        for L in range(0,len(net_size)):
            Nlay = []
            Alay = []
            for i in range(0,net_size[L]):
                Alay.append(0)
                if L > 0:
                    m = net_size[L - 1] # сколько нейронов на предыдущем слое столько здесь и весов
                    w = 2 * np.random.random((m, 1)) - 1
                    b = 0#np.random.random()
                    Nlay.append(Neuron(w,b,activ = activ))
            if L > 0:
                self.N.append(Nlay)
            self.A.append(Alay)

    def forward(self,x):
        for L in range(0,len(self.A)):
            for i in range(0,len(self.A[L])):
                if L == 0:
                    self.A[L][i] = x[i]
                else:
                    self.A[L][i] = self.N[L-1][i].activate(self.A[L - 1])

    def zL(self,L):
        zl = []
        for i in range(0,len(self.N[L])):
            zl.append(self.N[L][i].summatory(self.A[L]))
        return zl        

    def delta(self,L,y):#первый раз понадобился y
        Li = len(self.N)-1 # индекс последнего слоя
        n = len(self.N[Li])# длина последнего слоя
        if L == Li:
            delta = []
            for i in range(0,n):
                delta.append(self.cost_deriv(y,self.A[Li+1][i])*self.activ_deriv(self.zL(Li)[i]))
            return delta
        else:
            p = len(self.N[L])
            q = len(self.N[L+1])
            w_front = np.zeros((q,p))
            for j in range(0,q):
                for k in range(0,p):
                    w_front[j][k] = self.N[L+1][j].w[k].copy()
            delta_front = self.delta(L+1,y)
            delta_front = np.array(delta_front).reshape(len(delta_front),1)
            dot = w_front.T.dot(delta_front)
            F = self.zL(L)
            a = self.activ_deriv(np.array(F).reshape(len(F),1))
            delta = []
            for i in range(0,len(dot)):
                delta.append(dot[i]*a[i])
            return delta
        
    def setUpDeltas(self,x,y):
        self.forward(x)
        self.deltas = []
        for i in range(0,len(self.N)):
            self.deltas.append(self.delta(i,y))

    def train_on_single_example(self,x,y,learning_rate):
        self.setUpDeltas(x,y)
        for L in range(0,len(self.N)):
            for j in range(0,len(self.N[L])):
                for k in range(0,len(self.A[L])):
                    self.N[L][j].w[k] -= self.A[L][k]*self.deltas[L][j]*learning_rate
                #self.N[L][j].b -= self.deltas[L][j]*learning_rate

    def weightPrint(self):
        s = []
        for i in range(0,len(self.N)):
            l = []
            for j in range(0,len(self.N[i])):
                l.append(self.N[i][j].__str__())
            s.append(l)
        return s.__str__()
        

    def activPrint(self):
        s = []
        for i in range(0,len(self.A)):
            l = []
            for j in range(0,len(self.A[i])):
                l.append(self.A[i][j].__str__())
            s.append(l)
        return s.__str__()
      
n = 10
m = 5
X = 20 * np.random.sample((n, m)) - 10
y = (np.random.random(n) < 0.5).astype(np.int)[:, np.newaxis]

D = Discriminator(net_size = [5,1],activ = sigmoid,activ_deriv = sigmoid_prime,cost = quadratic,cost_deriv = quadratic_deriv)

for i in range(0,n):
    D.train_on_single_example(X[i].reshape(m,1),y[i],0.1)

'''
x1 = [1,0.3]#np.array([1,0.3]).reshape(2,1)
y1 = 1#np.array([[1]])
x2 = [0.4,0.5]#np.array([0.4,0.5]).reshape(2,1)
y2 = 1#np.array([[1]])
x3 = [0.7,0.8]#np.array([0.7,0.8]).reshape(2,1)
y3 = 0#np.array([[0]])

D.train_on_single_example(x1,y1,1)
D.train_on_single_example(x2,y2,1)
D.train_on_single_example(x3,y3,1)
'''
print(D.weightPrint())
print()
print(D.activPrint())

