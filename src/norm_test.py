
minibatch_size = 10.0
grads = [3.1,2.5,1.6,4.9]

val1 = sum([x**2 for x in grads])

val2 = sum([ (x/minibatch_size)**2 for x in grads ])

print "Stanford method",(val1**(0.5))/minibatch_size

print "My method",(val2**(0.5))