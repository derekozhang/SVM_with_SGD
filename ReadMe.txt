Name: Ulrik Soderstrom  Email: usoderst@u.rochester.edu Implement SGD for SVM for the adult income dataset. Experiment with performance as a function of learning rate.
 ************ Files ********* SVM.py is the code that implements the SVM with SGD. The code uses the data files a7a.dev, a7a.train, and a7a.test from the same directory (data files not included). The code requires a numpy and random import.

Accuracies_vs_eta.pdf is a graph resulting from the commented out function Graph_Acc_VS_eta that shows accuracies of the test set as a function of learning rates. The graph includes the following learning rates: [.005,.01,.02,.04,.05,.06,.07,.08,.09,.1,.2,.4,.6,.8,1]

ReadMe.txt is a text file that contains instructions and information about the code.  ************ Algorithm *****
 This algorithm is a support vector machine with stochastic gradient descent. The SVM trains on a training set and validates the best variables to yield the best performance (as discussed in Your Interpretation section). The SVM maximizes the margin between two hyperplanes. The margin M is equal to 2/||w||. This margin has flexibility as to how many points can sit inside it as allowed by C, a slack variable. 

Each update to the weights used to predict the classification of the testing set is based on the equation 1/N*w-Cynx(n) if 1-yn(WT*x(n)+b)>0 and 1/Nw otherwise. The bias variable b, is updated on the same criteria at Cyn or 0. The updates are done one vector at a time in SGD fashion. The validation set helps define the optimal parameters for C (slack), n (iterations), and Learning Rate (eta). 

Algorithm Steps:
1) Function file_input== reads in adult income data files and parses them into numpy arrays for testing, training, and validation sets.

2) Function Train_Model trains the weights based on the SVM update formula training vector by training vector in SGD fashion. The weights are updated by 1/N*w-Cynx(n) if 1-yn(WT*x(n)+b)>0 and 1/Nw otherwise. The bias variable b, is updated on the same criteria at Cyn or 0. The algorithm integrates 100 times. 

2) Function Validate_Model trains the weights in the same manner as Train_Model, but validates the best parameter functions (n, C, learning rate) to optimize performance. 

3) Function Test_Model then tests the resulting optimizes parameter weights on a test set. The accuracy/error rate is then reported. The accuracy can be graphed as a function of learning rates tried in the Graph_Acc_VS_eta function. 
 ************ Instructions *** 
To run the code, use python 2 or 3. The code pulls the data files: ‘a7a.dev’, ‘a7a.train’, and ‘a7a.test’ from the same directory/folder that the code file is in. 

There are two graph functions in the code that are commented out. Graph_Acc_VS_eta is a function that graphs each tested learning rate against the resulting accuracy on the test set. Graph_Learning_Rates graphs the errors over time at a given learning rate. In the folder, an example of Graph_Acc_VS_eta function’s output is included as Accuracies_vs_eta.pdf.

************ Results ******* 
The code performed with a top accuracy of ~82% at a learning rate of .04, iterations of 150, and slack variable of 10^-3.

Numerous learning rates of [.005,.01,.02,.04,.05,.06,.07,.08,.09,.1,.2,.4,.6,.8,1] were implemented. The accuracies ranged from high 70% to low 80%. The number of iterations did not have a significant affect when changed against these learning rates, generally iterations numbers between 50-200 were used. The slack hyper-parameter was tested with values ranging from 10^-10 to 100. The extreme mins and maxes in this range results in poor accuracies of ~25%. Any value from 10^-2 through 10^-6 allowed for maximum accuracies as a function of learning between 70% to low 80%. 

The included graph ‘Accuracies_vs_eta.pdf’ shows the different accuracies as a function of learning rate given n = 100 and C = 10^-3, their optimal levels. 

The code will have the following result in the terminal: 

Reading in and Preprocessing: a7a.dev
Reading in and Preprocessing: a7a.train
Reading in and Preprocessing: a7a.test

('Training Learning Rate at ', 0.4)
('Accuracy at this eta is ', 0.8278980891719745)

************ Your interpretation ****  
The hyper-paramters tuned include learning rate (eta), slack (C), and iterations (n). 

The range of slack values trained and tested were from 10^-10 and 100. Too large of a slack allows too many points to be in the margin while it is maximized. Likewise, too small of a slack value allows too few features to be included in the margin, shrinking the maximized margin between the two hyperplanes. A slack variable in the range 10^-2 to 10^-6 allowed for the most optimal results. 

The number of iterations used were between 50-200 and did not have a significant effect on accuracy. Generally, a iteration number of 100 was used. 

The learning rates had an important affect on accuracy. Numbers that were 10^-4 or lower resulted in poor accuracies. Likewise, accuracy drops off at high of a eta past ~.6. The optimal learning rate found is .4.


When implementing b, it is clear that most times it is zero, signaling that most of the hyperplanes are about the origin, and don’t need to be shifted. 

