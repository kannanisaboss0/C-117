#--------------------------------------------------------------ConfusionMatrix.py--------------------------------------------------------------#
'''
Importing modules:
-LogisticRegression (LogReg) :-sklearn.linear_model
-accuracy_score (a_s) :-sklearn.metrics
-StandardScaler (StandSC) :-sklearn.preprocessing
-confusion_matrix (conf_mat) :-sklearn.metrics
-matplotlib.pyplot (plt)
-pandas (pd)
-numpy (np)
-seaborn (sns)
-train_test_split (tts) :-sklearn.model_selection
-time (tm)
'''

from sklearn.linear_model import LogisticRegression as LogReg
from sklearn.metrics import accuracy_score as a_s
from sklearn.preprocessing import StandardScaler as StandSc
from sklearn.metrics import confusion_matrix as conf_mat
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split as tts
import time as tm


#Reading data from the file
df=pd.read_csv("data.csv")

#Introductory staement and inputs
print("Welcome to ConfusionMatrix.py. We provide statistical data about diabetic probability.")

view_information=input("Do not know about confusion matrix?(:-I Know, I Don't Know)")

#Verifying whether the user wants to view information about confusion matrix
#Case-1
if(view_information=="I Don't Know" or view_information=="I don't Know" or view_information=="I Don't know" or view_information=="i don't know"):
  print("What is a confusion matrix?")
  tm.sleep(2.3)

  print("In Logistic Regression, espically classififcation, the  accuracy of the predication may not always be accurate.")
  tm.sleep(3.4)

  print("Sometimes, incorrect predictions occur and are often difficult to locate.")
  tm.sleep(1.2)

  print("Hence, in order to facilitate this difficulty, confusion matrixes are used.")
  tm.sleep(0.2)

  print("Confusion matrixes describe the summary of the the prediction, precisely, the incorrect and correct predictions")
  tm.sleep(4.5)

  print("Layout:")
  tm.sleep(2.9)

  print("A confusion matrix is illustrated in the form of a table, usually with dimensions 2 by 2.")
  tm.sleep(3.4)

  print("The rows are occupied by the classes, namely the predicted class and actual class.")
  print("The columns are occupied by instances of each class.")
  tm.sleep(3.9)

  print("The correct prediction and actual value are always arranged vertically oppositive to each other.")
  print("The incorrect prediction and actual value are always arranged vertically oppositive to each other.")
  print("Errors are differentiated and hence, it can also be called error matrix")
  tm.sleep(4.6)
  
  print("The accuracy percentage of the classifier can also be calculated with the derivation:")
  tm.sleep(0.2)

  print("        (sum of correct predictions)")
  print("        ____________________________               x100")
  print("(sum of all values including correct predictions)")
  tm.sleep(2.3)

  print("To know more about Confusion Matrix, visit 'https://en.wikipedia.org/wiki/Confusion_matrix'")
  tm.sleep(2.3)

#Case-2
else:
  print("Request Accepted")
  tm.sleep(1.3)

print("Loading Data...")
tm.sleep(2.1)

number=int(input("Enter the number of statistics desired to predict the probability of the patient accquiring diabetes[The number should range between 1 and 6]:"))

#Verifying whether the number provided does not exceed the maximum number or minimum number of statistics
#Case-1
if(int(number)<=6 and int(number)>=1):
  stat_list=["Unusable_Element","Glucose","Blood Pressure","Skin Thickness","Insulin Levels","BMI","Age"]
  count_stat=0

  for stat in stat_list[1:]:
    count_stat+=1
    print("{}:{}".format(count_stat,stat)) 

  train_list=[]  
  count_loop=1

  while (count_loop<(int(number)+1)):
    stat_input=int(input("Please enter the corresponding index of the statistics desired to predict with:(statistic number {})".format(count_loop)))
    stat_choice=stat_list[stat_input]

    for value in train_list:

      #Verifying whether a statistic has been not repeated in the list of selected statistics or a statistic has been repeated
      #Case-1
      if(value!=stat_choice):
        continue

      #Case-2
      else:
        print("Request Terminated.")
        print("Invalid Input.")
        print("Each value provided should be unique to each other and not be repetitive.")
        print("Thank you for using ConfusionMatrix.py")
        break

    train_list.append(stat_choice) 
    count_loop+=1

  x_value=''
  y_value=''

  axis_list=["Unusabel_Element","Predicted Values","Actual Values"]
  axis_count=0

  for axis_value in axis_list[1:]:
    axis_count+=1

    print("{}:{}".format(axis_count,axis_value))

  axis_input=int(input("Please enter the corresponding index of the value desired to be the x-axis:"))
  axis_choice=axis_list[axis_input]

  #Assessing the user's decision on wihc class should be the x-axis
  #Case-1
  if(axis_input==1):
    x_value=prediction_list
    x_label="Predicted Data"

    y_value=actual_list
    y_label="Actual Data"

  #Case-2
  elif(axis_input==2):
    y_value=prediction_list
    y_label="Predicted Data"

    x_value=actual_list
    x_label="Actual Data"

  labels=["Yes","No"]

  c_m=conf_mat(x_value,y_value,labels)

  sub_plot=plt.subplot()



  heat_map=sns.heatmap(c_m,annot=True,ax=sub_plot)

  sub_plot.set_ylabel(x_label)
  sub_plot.set_xlabel(y_label)

  sub_plot.set_title("Diabetes Probability:- Heat Map (Confusion Matrix)")

  sub_plot.xaxis.set_ticklabels(labels)
  sub_plot.yaxis.set_ticklabels(labels)

  plt.show()

  c_m=c_m.ravel()

  confusion_matrix_accuracy=(c_m[0]+c_m[3])/(c_m[0]+c_m[1]+c_m[2]+c_m[3])

  c_m_percentage=round(confusion_matrix_accuracy*100,2)

  print("The veracity of the data is {}%".format(c_m_percentage))

  #Prinitng ending message
  print("Thank You for using ConfusionMatrix.py")

#Case-2
elif(int(number)>6):
  print("Request Terminated.")
  print("Invalid Input.")
  print("The number should be lesser than or equal to 6.")
  
  #Prinitng ending message
  print("Thank You for using ConfusionMatrix.py")

#Case-3
elif(int(number)<1):
  print("Request Terminated.")
  print("Invalid Input.")
  print("The number should be greater than or equal to 1.")
  
  #Prinitng ending message
  print("Thank You for using ConfusionMatrix.py")

#Case-4
else:
  print("Request Terminated.")
  print("Invalid Input.")
  print("The value provided should be a whole number between 0 and 7, extremes not considered.")

  #Prinitng ending message
  print("Thank You for using ConfusionMatrix.py")


df_train=df[train_list]
df_result=df["Diabetes"]

factor_train,factor_test,result_train,result_test=tts(df_train,df_result,test_size=0.25,random_state=0)

lr=LogReg(random_state=0)
SS=StandSc()



factor_train=SS.fit_transform(factor_train)
factor_test=SS.fit_transform(factor_test)

lr.fit(factor_train,result_train)

predictions=lr.predict(factor_test)

predictions=predictions.ravel()


actual_list=[]
for actual in result_test:

  #Assessing the binary value and append according values to the list, where 0 is "No" and 1 is "Yes"
  #Case-1
  if(actual==0):
    actual_list.append("No")

  #Case-2
  else:
    actual_list.append("Yes")  

prediction_list=[]
for prediction in predictions:

  #Assessing the binary value and append according values to the list, where 0 is "No" and 1 is "Yes"
  #Case-1
  if(prediction==0):
    prediction_list.append("No")

  #Case-2
  else:
    prediction_list.append("Yes")   



#--------------------------------------------------------------ConfusionMatrix.py--------------------------------------------------------------#
