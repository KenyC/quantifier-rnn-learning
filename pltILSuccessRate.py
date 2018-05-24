import numpy as np 
import matplotlib.pyplot as plt
import csv

EXPES = [
	{"folder": "thefirstthree/", "trials": [0,2,4]}, 
	{"folder": "atleastthree/", "trials": range(3)}]
NGEN = 40
COLORS = ["red", "green"]


# DATA_DIR = "atleastthree/"
# NUM_TRIALS = 3

def getResult(file_path):
	with open(file_path,"r") as f:
		reader = csv.DictReader(f)
		return [row for row in reader]

def getExpe(expe, selector="Actual success rate"):
	
	data = [getResult(expe["folder"]+"trial"+str(i)+"/results.csv") for i in expe["trials"]]
	
	return np.array([[float(row[selector]) for row in d] for d in data])

def getAvgExpe(expe, selector="Actual success rate"):
	return np.mean(getExpe(expe,selector), axis = 0)

def plotAxes(ax, selector):
	x = np.arange(NGEN)
	ax.set_title(selector)
	for i,exp in enumerate(EXPES):
		res = getExpe(exp, selector)
		ax.plot(x, np.transpose(res), color = COLORS[i])
		#ax.plot(np.arange(10))#, color = COLORS[i])


fig, axes = plt.subplots(2,2)



plotAxes(axes[0,0],"z-score")
plotAxes(axes[1,0],"Actual success rate")
plotAxes(axes[1,1],"p(true)")

# for i in range(NUM_TRIALS):
# 	plt.plot(generation[0],zscore[i,], label = "trial n{}".format(i))

#plt.legend()



# for d in data:
# 	plt.plot([row["generation"] for row in d],[row["z-score"] for row in d])

plt.show()
# plt.exit()