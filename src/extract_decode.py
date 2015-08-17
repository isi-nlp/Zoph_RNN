
data = [line.replace('\n','') for line in open('train_2.txt')]

output = file('decode.txt','w')

bad = 
for i in range(0,len(data),4):
	data[i] = data[i].split(' ')

	data[i] = [x for x in data[i] if x !='-1']
	output.write(data[i] +'\n')