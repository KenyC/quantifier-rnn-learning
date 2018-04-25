import os
import csv
from collections import defaultdict
import matplotlib.pylab as plt

csvDir = "data/expKeny/expBiased"


quantifierColors = dict(not_all = "r",not_only = "b")




plt.figure(0)


# with open(csvDir+"trial_1.csv") as csvfile:
# 	reader = csv.DictReader(csvfile)
# 	xs = []
# 	ys = defaultdict(list)
# 	for row in reader:

# 		xs.append(row["global_step"])
		
# 		for q in quantifierColors.keys():
# 			ys[q].append(row[q+"_accuracy"])
# 	for key in ys:
# 		plt.plot(xs,ys[key],quantifierColors[key])

for file in os.listdir(csvDir):
	if file.endswith(".csv"):
		pathCsvFile = os.path.join(csvDir, file)
		
		with open(pathCsvFile) as csvfile:
			reader = csv.DictReader(csvfile)
			xs = []
			ys = defaultdict(list)
			for row in reader:

				xs.append(row["global_step"])
				
				for q in quantifierColors.keys():
					ys[q].append(row[q+"_accuracy"])
			for key in ys:
				plt.plot(xs,ys[key],quantifierColors[key])




plt.show()
plt.close()