from math import factorial

def combination(n, r):
	return factorial(n)/(factorial(r) * factorial(n - r))

def get_prob(N):
	# N is number of nodes
	le_six_prob = [combination(N, i) * ((1 / N)**i) * ((1 - (1 / N))**(N - i)) for i in range(7)]
	return 1 - sum(le_six_prob)

print(f'maximum > 6 collision probability is {max([get_prob(N) for N in range(7, 1023)])} in the range (6, 1023)')
