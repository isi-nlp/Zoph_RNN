import sys
import re
import numpy as np
import pandas as pd

def parse(fn,key = "all", side = 0, the_layer = 0):

    f = open(fn)

    ys = []
    cts = []
    hts = []
    forget_gates = []
    input_gates = []
    output_gates = []

    p = re.compile(r'-+Layer[ ]?([0-9]+)[ ]?Source word: ([0-9]+)-+')
    if side == 1:
        p = re.compile(r'-+Layer[ ]?([0-9]+)[ ]?Target word: ([0-9]+)-+')

    while True:
        line = f.readline()
        if not line:
            break
        line = line.strip()
        m = p.match(line)

        if m:
            layer = int(m.groups()[0]) - 1
            if layer >= len(ys):
                ys.append([])
                cts.append([])
                hts.append([])
                forget_gates.append([])
                input_gates.append([])
                output_gates.append([])


            index = int(m.groups()[1])
            ys[layer].append(index+1)
            # vocab index
            if layer == 0:
                line = f.readline()
            # forget gate
            line = f.readline()
            if key == "fg" or key == "all":
                line = line.split(':')[1]
                ll = [float(x) for x in line.split()]
                forget_gates[layer].append(ll)
            # input gate
            line = f.readline()
            if key == 'ig' or key == "all":
                line = line.split(':')[1]
                ll = [float(x) for x in line.split()]
                input_gates[layer].append(ll)
            # c_t
            line = f.readline()
            if key == 'ct' or key == "all":
                line = line.split(':')[1]
                ll = [float(x) for x in line.split()]
                cts[layer].append(ll)
            # output gate
            line = f.readline()
            if key == 'og' or key == "all":
                line = line.split(':')[1]
                ll = [float(x) for x in line.split()]
                output_gates[layer].append(ll)
            # h_t
            line = f.readline()
            if key == 'ht' or key == "all":
                line = line.split(':')[1]
                ll = [float(x) for x in line.split()]
                hts[layer].append(ll)        
            
    
    def convert_np(a):
        npa = []
        for aa in a:
            npa.append(np.array(aa))
        return npa
                
    npys = convert_np(ys)
    npcts = convert_np(cts)
    nphts = convert_np(hts)
    npfgs = convert_np(forget_gates)
    npigs = convert_np(input_gates)
    npogs = convert_np(output_gates)

    l = the_layer
    return (npys[l], npcts[l], nphts[l], npfgs[l], npigs[l], npogs[l])

#parse('./lstm.txt')

def split_into_sentece(res):
    # res = (npys[l], npcts[l], nphts[l], npfgs[l], npigs[l], npogs[l])

    y = res[0]
    res_sent = []
    for i in xrange(len(res)):
        res_sent.append([])
        
    for i in xrange(len(y)):
        if y[i] == 1:
            for j in xrange(len(res)):
                if len(res[j]) > 0:
                    res_sent[j].append([])
        for j in xrange(len(res)):
            if (len(res[j])) > 0:
                res_sent[j][-1].append(res[j][i])
            
    return res_sent

def convert_to_hdf(lstm_path,en_path,hdf_path):
    nsent = len(res[0])
    d = {}
    for sid in xrange(nsent):
        sid in xrange(nsent)
