import numpy as np 
import matplotlib.pyplot as plt
import csv

DATA_DIR = "thefirstthree/"
NUM_TRIALS = 5

# DATA_DIR = "atleastthree/"
# NUM_TRIALS = 3

def getResult(file_path):
	with open(file_path,"r") as f:
		reader = csv.DictReader(f)
		return [row for row in reader]

data = [getResult(DATA_DIR+"trial"+str(i)+"/results.csv") for i in range(NUM_TRIALS)]
generation = [[row["generation"] for row in d] for d in data]
zscore = np.array([[float(row["Actual success rate"]) for row in d] for d in data])

for i in range(NUM_TRIALS):
	plt.plot(generation[0],zscore[i,], label = "trial n{}".format(i))

plt.legend()



# for d in data:
# 	plt.plot([row["generation"] for row in d],[row["z-score"] for row in d])

plt.show()
# plt.exit()