# python read_hpc_output.py folder 

import sys
import os
import pandas as pd

def parse_arg():
    d = {}
    d['folder'] = sys.argv[1]
    return d

def process_file(path):
    f = open(path)
    d = {}
    d['epoch'] = 1
    d['dev'] = 0.0
    d['train'] = 0.0
    for line in f:
        ll = line.split()
        if line.startswith("New dev set Perplexity:"):
            d['dev'] = float(ll[-1])
        if line.startswith("Training set perplexity:"):
            d['train'] = float(ll[-1])
        if line.startswith("Starting epoch"):
            d['epoch'] = int(ll[-1])
    f.close()
    return d
    

def main():
    args = parse_arg()

    table = {}

    for fn in os.listdir(args['folder']):
        if fn.startswith("HPC_OUTPUT"):
            if not fn.endswith("decode"):
                row = process_file(os.path.join(args['folder'],fn))
                key = fn[11:]
                table[key] = row

    # print the row
    df = pd.DataFrame.from_dict(table,orient = "index")
    
    print df
    
if __name__ == '__main__':
    main()
