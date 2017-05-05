import random

def generate(word,start,end,i,f):
    s = start
    n = i
    for l, w in enumerate(word):
        if l == len(word) -1:
            n = end
        f.write("({} ({} {}))\n".format(s,n,w))
        s = n
        i += 1
        n = i
    return i

words = ["lstm","is","great","slow","luckily","we","make","it","fast","enough","and","with","fsa"]

def generate_fsa():
    f = open('fsa.txt','w')
    f.write("E\n")

    i = 1
    for word in words:
        i = generate(word,"S","E",i,f)

    f.write("(E (S *e*))\n")
    f.close()

def generate_pair():
    s = ""
    for i in xrange(3):
        s += random.choice(words)
    target = " ".join(s)
    source = []
    for w in s:
        n = ord(w) % 13
        source.append(n)
    source = " ".join([str(x) for x in source])
    return source, target

def generate_fn_pairs(n,fn1,fn2):
    f1 = open(fn1,'w')
    f2 = open(fn2,'w')
    for i in xrange(n):
        source, target = generate_pair()
        f1.write(source + "\n")
        f2.write(target + "\n")
    f1.close()
    f2.close()

def generate_train_valid():
    generate_fn_pairs(1000,"source.train.txt","target.train.txt")
    generate_fn_pairs(100,"source.valid.txt","target.valid.txt")
    generate_fn_pairs(1,"source.test.txt","target.test.txt")

generate_fsa()
generate_train_valid()

