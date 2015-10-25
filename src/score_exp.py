import numpy as np

D = 10
p_t = 12.5
S = [12,13,14]

denom = (D**2)/4.0


#exp( ( -1*pow_wrapper( ( d_p_t[minibatch_index] - d_indicies[minibatch_index + minibatch_size*i] ) ,2.0) )/(2*sigma_sq) );


for s in S:
	val = np.e**( (-(s-p_t)**2)/(2*denom))
	print "p_t=",p_t,"    s=",s,"    val =",val





#   0.606003105640411 
#   0.607058167457581 
#   0.0111380163580179 