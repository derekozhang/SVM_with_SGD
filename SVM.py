import numpy as np 
import random
import matplotlib.pyplot as plt

'''Ulrik Soderstrom
SVM with SGD 
'''

#Reading in adult income data set 
def file_input(filename):
	print("Reading in and Preprocessing: " + filename)
	weights_list_temp = []
	label_list_temp = []
	with open(filename, "r") as Data_File:
		for line in Data_File:
			temp_line_list = []
			line_list = [x.strip() for x in line.lower().split(" ")]
			if len(line_list) == 16:
				del line_list[-1]
				label = int(line_list.pop(0))
#				if label == -1:
#					label = 0
				for item in line_list:
					item = item.replace(':','')
					item = item[:-1]
					item = int(item)
					temp_line_list.append(item)
				weights_list_temp.append(temp_line_list)
				label_list_temp.append(label)
		weights_np = np.array(weights_list_temp)
		label_np = np.array(label_list_temp)
	return weights_np, label_np

Dev_array, Dev_x = file_input("a7a.dev")

Train_array, Train_x = file_input("a7a.train")

Test_array, Test_x = file_input("a7a.test")

print("\n")

#Training Model 
def Train_Model(w,Train_array,Train_x,Learning_rate,C,b):
	N = float(len(Train_array))
	for i in range(0,len(Train_array)):
		result = 1 - Train_x[i] * np.dot(w,Train_array[i]+b)
		if result > 0:
			w = w - 1 * Learning_rate * C * Train_x[i] * Train_array[i]+b
			b = b - C * Train_x[i]
		else:
			w = w
			b = b
	return w, b

#Validating Model
def Validate_Model(w,Dev_array,Dev_x,Learning_rate,C,b):
	N = float(len(Train_array))
	for i in range(0,len(Train_array)):
		result = 1 - Train_x[i] * np.dot(w,Train_array[i]+b)
		if result > 0:
			w = w - 1 * Learning_rate * C * Train_x[i] * Train_array[i]+b
			b = b - C * Train_x[i]
		else:
			w = w
			b = b
	return w, b


#Testing Model and returning accuracy
def Test_Model(w,Test_array,Test_x,Learning_rate,C,b):
	error_count = 0
	for i in range(len(Test_array)):
		result = 1 - Test_x[i] * np.dot(w,Test_array[i]+b)
		if result > 0:
			error_count += 1
		else:
			error_count += 0
	return (1-(float(error_count)/float((len(Test_array)+1))))

#Graphs Learning Rate
def Graph_Learning_Rate(errors):
	num = 0
	den = 0
	count = 0
	y = []
	x = []
	for i in range(len(errors)):
		den += 1
		count += 1
		if errors[i] == 0:
			num += 1
		else:
			num = num
		y.append(float(num)/float(den))
		x.append(count)
	plt.plot(x,y)
	plt.title('Perceptron Learning Rate')
	plt.xlabel('time')
	plt.ylabel('error rate')
	plt.grid(True)
	plt.show()

#Graphs Accuracies vs Learning Rates
def Graph_Acc_VS_eta(Accuracies,Learning_rates):
	plt.plot(Learning_rates,Accuracies,linestyle='--',marker='D',color='b')
	plt.title('Accuracy versus Learning Rate')
	plt.xlabel('Learning Rates')
	plt.ylabel('Accuracies')
	plt.grid(True)
	plt.savefig("Accuracies_vs_eta.pdf")
	plt.show()

#Variables for Model
errors = []
Accuracies = []
n = 100
C = 10^-3


#Looping to train model
print("Training Models at Different Learning Rates")
#Learning_rates = [.005,.01,.02,.04,.05,.06,.07,.08,.09,.1,.2,.4,.6,.8,1]
Learning_rates = [.4,]
for eta in Learning_rates:
	print("Training Learning Rate at ",eta)
	w = np.random.rand(14)
	b = 0
	for i in range(n):
		w, b = Train_Model(w,Train_array,Train_x,eta,C,b)

	#print("Validating Model")
	for i in range(n):
		w, b = Validate_Model(w,Dev_array,Dev_x,eta,C,b)
	
	Acc = (Test_Model(w,Test_array,Test_x,eta,C,b))
	print("Accuracy at this eta is ", Acc)
	print("\n")
	Accuracies.append(Acc)

#Graph_Acc_VS_eta(Accuracies,Learning_rates)
#Graph_Learning_Rate(errors)


