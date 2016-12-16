import sys
for line in sys.stdin:
    ll = line.split()
    for letter in ll:
        print ord(letter) % 10,
    print
