#load the non attentional model 

import numpy as np
from cStringIO import StringIO
from datetime import datetime
from numpy import linalg

def read_matrix(row,column,f):
    start = datetime.now()
    m = np.zeros((row,column))
    for i in xrange(row):
        line = f.readline()
        ll = [float(x) for x in line.split()]
        for j in xrange(len(ll)):
            m[i,j] = ll[j]
    # read the empty line
    f.readline()
    end = datetime.now()
    print m.shape, end-start
    return m

def write_matrix(m,f):  
    start = datetime.now()      
    row = m.shape[0]
    for i in xrange(row):
        line = " ".join([str(x) for x in m[i]]) + "\n"
        f.write(line)
    # write an empty_line
    end = datetime.now()
    print m.shape, end-start
    
    f.write("\n")

def same_matrix(m1,m2):
    assert(m1.shape == m2.shape)
    nm = linalg.norm(m1 - m2)
    print nm
    if nm < 1e-6:
        return True
    else:
        return False

def random_matrix(row,column,upper = 0.08):
    m = (np.random.rand(row,column) - 0.5) / 0.5 * upper
    return m

class IH:
    def __init__(self,LSTM_size, vocab_size):
        self.LSTM_size = LSTM_size
        self.vocab_size = vocab_size
        self.parameter_names = ["w_hi","b_i","w_hf","b_f","w_hc","b_c","w_ho","b_o", "w", "m_i","m_f","m_o","m_c"]

    def parse(self,f):
        self.w_hi = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.b_i = read_matrix(self.LSTM_size,1,f)
        self.w_hf = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.b_f = read_matrix(self.LSTM_size,1,f)
        self.w_hc = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.b_c = read_matrix(self.LSTM_size,1,f)
        self.w_ho = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.b_o = read_matrix(self.LSTM_size,1,f)
        
        self.w = read_matrix(self.LSTM_size,self.vocab_size,f)
        self.m_i = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.m_f = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.m_o = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.m_c = read_matrix(self.LSTM_size,self.LSTM_size,f)

    def dump(self,f):
        write_matrix(self.w_hi,f)
        write_matrix(self.b_i,f)
        write_matrix(self.w_hf,f)
        write_matrix(self.b_f,f)
        write_matrix(self.w_hc,f)
        write_matrix(self.b_c,f)
        write_matrix(self.w_ho,f)
        write_matrix(self.b_o,f)

        write_matrix(self.w,f)
        write_matrix(self.m_i,f)
        write_matrix(self.m_f,f)
        write_matrix(self.m_o,f)
        write_matrix(self.m_c,f)
        

    def random_weight(self):
        self.w_hi = random_matrix(self.LSTM_size,self.LSTM_size)
        self.b_i = random_matrix(self.LSTM_size,1)
        self.w_hf = random_matrix(self.LSTM_size,self.LSTM_size)
        self.b_f = np.ones((self.LSTM_size,1))
        self.w_hc = random_matrix(self.LSTM_size,self.LSTM_size)
        self.b_c = random_matrix(self.LSTM_size,1)
        self.w_ho = random_matrix(self.LSTM_size,self.LSTM_size)
        self.b_o = random_matrix(self.LSTM_size,1)
        
        self.w = random_matrix(self.LSTM_size,self.vocab_size)
        self.m_i = random_matrix(self.LSTM_size,self.LSTM_size)
        self.m_f = random_matrix(self.LSTM_size,self.LSTM_size)
        self.m_o = random_matrix(self.LSTM_size,self.LSTM_size)
        self.m_c = random_matrix(self.LSTM_size,self.LSTM_size)
        
    def is_same(self,other):
        same = False;
        print "===== IH ====="
        for name in self.parameter_names:
            m1 = getattr(self,name)
            m2 = getattr(other,name)
            s = same_matrix(m1,m2)
            same = same and s
            print name, s
        return same

        
        

class HH:
    def __init__(self,LSTM_size):
        self.LSTM_size = LSTM_size
        self.parameter_names = ["w_hi","b_i","w_hf","b_f","w_hc","b_c","w_ho","b_o", "m_i","m_f","m_o","m_c"]

    def parse(self,f):
        self.w_hi = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.b_i = read_matrix(self.LSTM_size,1,f)
        self.w_hf = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.b_f = read_matrix(self.LSTM_size,1,f)
        self.w_hc = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.b_c = read_matrix(self.LSTM_size,1,f)
        self.w_ho = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.b_o = read_matrix(self.LSTM_size,1,f)
        
        self.m_i = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.m_f = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.m_o = read_matrix(self.LSTM_size,self.LSTM_size,f)
        self.m_c = read_matrix(self.LSTM_size,self.LSTM_size,f)
        
    def random_weight(self):
        self.w_hi = random_matrix(self.LSTM_size,self.LSTM_size)
        self.b_i = random_matrix(self.LSTM_size,1)
        self.w_hf = random_matrix(self.LSTM_size,self.LSTM_size)
        self.b_f = np.ones((self.LSTM_size,1))
        self.w_hc = random_matrix(self.LSTM_size,self.LSTM_size)
        self.b_c = random_matrix(self.LSTM_size,1)
        self.w_ho = random_matrix(self.LSTM_size,self.LSTM_size)
        self.b_o = random_matrix(self.LSTM_size,1)
        
        self.m_i = random_matrix(self.LSTM_size,self.LSTM_size)
        self.m_f = random_matrix(self.LSTM_size,self.LSTM_size)
        self.m_o = random_matrix(self.LSTM_size,self.LSTM_size)
        self.m_c = random_matrix(self.LSTM_size,self.LSTM_size)

    def dump(self,f):
        write_matrix(self.w_hi,f)
        write_matrix(self.b_i,f)
        write_matrix(self.w_hf,f)
        write_matrix(self.b_f,f)
        write_matrix(self.w_hc,f)
        write_matrix(self.b_c,f)
        write_matrix(self.w_ho,f)
        write_matrix(self.b_o,f)

        write_matrix(self.m_i,f)
        write_matrix(self.m_f,f)
        write_matrix(self.m_o,f)
        write_matrix(self.m_c,f)

    def is_same(self,other):
        same = False;
        print "===== HH ====="
        for name in self.parameter_names:
            m1 = getattr(self,name)
            m2 = getattr(other,name)
            s = same_matrix(m1,m2)
            same = same and s
            print name, s
        return same


class Softmax:
    def __init__(self,LSTM_size, vocab_size):
        self.LSTM_size = LSTM_size
        self.vocab_size = vocab_size
        self.parameter_names = ["D","b"]

    def parse(self,f):
        self.D = read_matrix(self.vocab_size,self.LSTM_size,f)
        self.b = read_matrix(self.vocab_size,1,f)

    def random_weight(self):
        self.D = random_matrix(self.vocab_size, self.LSTM_size)
        self.b = random_matrix(self.vocab_size, 1)

    def dump(self,f):
        write_matrix(self.D,f)
        write_matrix(self.b,f)
        
    def is_same(self,other):
        same = False;
        print "===== softmax ====="
        for name in self.parameter_names:
            m1 = getattr(self,name)
            m2 = getattr(other,name)
            s = same_matrix(m1,m2)
            same = same and s
            print name, s
        return same


class Model:
    def __init__(self):
        self.LM = False;
        self.source_vocab = {}
        self.target_vocab = {}
        self.source_size = 0
        self.target_size = 0
        self.num_layers = 1
        self.LSTM_size = 1000
        self.source_layers = []
        self.target_layers = []
        self.softmax = None
    
    def diff(self,other):

        for i in xrange(len(self.source_layers)):
            self.source_layers[i].is_same(other.source_layers[i])

        for i in xrange(len(self.target_layers)):
            self.target_layers[i].is_same(other.target_layers[i])
            
        self.softmax.is_same(other.softmax)

    
    def random_model(self,num_layers,LSTM_size, source_size, target_size):
        self.LM = False;
        self.source_size = source_size
        self.target_size = target_size
        self.num_layers = num_layers
        self.LSTM_size = LSTM_size
        
        for i in xrange(self.source_size):
            self.source_vocab[i] = str(i)
        for i in xrange(self.target_size):
            self.target_vocab[i] = str(i)
        
        self.source_layers.append(IH(LSTM_size,source_size))
        for i in xrange(1,num_layers):
            self.source_layers.append(HH(LSTM_size))

        self.target_layers.append(IH(LSTM_size,target_size))
        for i in xrange(1,num_layers):
            self.target_layers.append(HH(LSTM_size))
            
        self.softmax = Softmax(LSTM_size, target_size)

        for layer in self.source_layers + self.target_layers:
            layer.random_weight()

        self.softmax.random_weight()


    def load_vocab(self,side,f):
        while True:
            line = f.readline()
            if line.startswith("==="):
                break
            ll = line.split()
            index = int(ll[0])
            word = ll[1]
            if side == 0:
                self.source_vocab[index] = word
            elif side == 1:
                self.target_vocab[index] = word
        if side == 0:
            print "Finish loading source vocab"
        else:
            print "Finish loading target vocab"
            
    def write_vocab(self,side,f):
        vocab = self.target_vocab
        vs = self.target_size
        if side == 0:
            vs = self.source_size
            vocab = self.source_vocab
        f.write("==========================================================\n")
        for i in xrange(vs):
            line = unicode(i) + u" " + vocab[i].decode('utf8') + u"\n"
            line = line.encode("utf8")
            f.write(line)

    def parse(self,fn):
        f = open(fn)
        line = f.readline()
        ll = line.split()
        if len(ll) == 3:
            self.LM = True
        self.num_layers = int(ll[0])
        self.LSTM_size = int(ll[1])
        self.target_size = int(ll[2])
        if not self.LM:
            self.source_size = int(ll[3])
        
        f.readline()

        # load source vocabs
        if not self.LM:
            self.load_vocab(0,f)
        
        # load target vocabs
        self.load_vocab(1,f)
            
        # source side
        if not self.LM:
            ih = IH(self.LSTM_size,self.source_size)
            ih.parse(f)
            self.source_layers.append(ih)
            for i in xrange(1,self.num_layers):
                hh = HH(self.LSTM_size)
                hh.parse(f)
                self.source_layers.append(hh)

        # target side
        ih = IH(self.LSTM_size,self.target_size)
        ih.parse(f)
        self.target_layers.append(ih)
        for i in xrange(1,self.num_layers):
            hh = HH(self.LSTM_size)
            hh.parse(f)
            self.target_layers.append(hh)

        # softmax
        softmax = Softmax(self.LSTM_size,self.target_size)
        softmax.parse(f)
        self.softmax = softmax

        f.close()

    def dump(self,fn):
        f = open(fn,"w")
        line = ""
        if self.LM: 
            line = "{} {} {}\n".format(self.num_layers,self.LSTM_size,self.target_size)
        else:
            line = "{} {} {} {}\n".format(self.num_layers,self.LSTM_size,self.target_size,self.source_size)        
        
        f.write(line)
        #write source vocabs
        if not self.LM:
            self.write_vocab(0,f)
        
        #write target vocabs
        self.write_vocab(1,f)
        
        f.write("==========================================================\n")
        # source side
        if not self.LM:
            for layer in self.source_layers:
                layer.dump(f)
                
        # target side
        for layer in self.target_layers:
            layer.dump(f)
            
        self.softmax.dump(f)
        
        f.close()

        
