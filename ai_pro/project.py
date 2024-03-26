import pandas as pd
import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split
from tabulate import tabulate

# Load CSV data
data = pd.read_csv("train.csv") #reads the CSV file and store it in Dta
data.drop("CoapplicantIncome", axis=1, inplace=True)

 #list of column names to be checked for missing values.
listt=["Gender","Married","Dependents","Education","Self_Employed","ApplicantIncome"
       ,"LoanAmount","Loan_Amount_Term","Credit_History","Property_Area"] 
            
for i in listt:
    data[i] = data[i].fillna(data[i].mode()[0]) # fill the empty value in the column by by most value common

# print(data.isnull().sum())  # Count missing values in each column

value_to_check= data[data.columns[1:10]].values #the checking coloumn
output_of_value = data[data.columns[11]].values # the results
#split 0.22 into test and the rest for train
X_train,X_test,Y_train,Y_test=train_test_split(value_to_check,output_of_value,test_size=0.22,shuffle=False) #split the data into train and test

## start encoding the Value of input and outout
from sklearn.preprocessing import LabelEncoder
combined_data = np.concatenate((X_train, X_test)) #to ensure that every value still the same in train and test.
LabelEncoder_X=LabelEncoder()


for i in range(0,9):
    combined_data[:, i] = LabelEncoder_X.fit_transform(combined_data[:, i]) # correct the value of each cell in train with the same in test
LabelEncoder_Y=LabelEncoder()

Y_train=LabelEncoder_Y.fit_transform(Y_train)  #encoder the output of train 
X_train = combined_data[:len(X_train), :] #select data from the began until arrive the end of legnth of X_train
X_test = combined_data[len(X_train):, :] #select data from the end of X_train to end(X_test)
Y_test=LabelEncoder_Y.fit_transform(Y_test) #encoder the output of test

from sklearn.neural_network import MLPClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------------
from sklearn.tree import DecisionTreeClassifier
Decision_tree=DecisionTreeClassifier(criterion='entropy', random_state=0) # creating a Decision Tree Classifier
 #decision tree will use information gain to make decisions during training. ---> 'entropy'
 #get the same set of random numbers -->random_state=0
Decision_tree.fit(X_train,Y_train) #fit model in training data (تدريب)
predict_of_Decision=Decision_tree.predict(X_test)  #pridiction on test data (تطبيق)
#---------------------------------------------------------------------------------------------------

from sklearn import metrics #to orint it in matrix
r=metrics.recall_score(predict_of_Decision,Y_test)
p=metrics.precision_score(predict_of_Decision,Y_test)
acc=metrics.accuracy_score(predict_of_Decision,Y_test)
f1=metrics.f1_score(predict_of_Decision,Y_test)
mat=metrics.confusion_matrix(predict_of_Decision,Y_test)
tn,fp,fn,tp=mat.ravel()
#_______________________________________________________________________________________________

from sklearn.naive_bayes import GaussianNB
naive_bayess=GaussianNB()  #creating a naive_bayes
naive_bayess.fit(X_train,Y_train) #fit model in training data (تدريب)
predict_naive_bayes=naive_bayess.predict(X_test) #pridiction on test data (تطبيق)

#_____________________________________________________________________________________________

r_naive=metrics.recall_score(predict_naive_bayes,Y_test)
p_naive=metrics.precision_score(predict_naive_bayes,Y_test)
acc_naive=metrics.accuracy_score(predict_naive_bayes,Y_test)
f1_naive=metrics.f1_score(predict_naive_bayes,Y_test)
mat1=metrics.confusion_matrix(predict_naive_bayes,Y_test)
tn_naive,fp_naive,fn_naive,tp_naive=mat1.ravel()
#________________________________________________________________________________________________
from sklearn.ensemble import RandomForestClassifier
random_foresrt = RandomForestClassifier(n_estimators=100, random_state=0)
random_foresrt.fit(X_train, Y_train)
predict_random = random_foresrt.predict(X_test)
#___________________________________________________________________________________________________
r_RF = metrics.recall_score(predict_random, Y_test)
p_RF = metrics.precision_score(predict_random, Y_test)
acc_RF = metrics.accuracy_score(predict_random, Y_test)
f1_RF = metrics.f1_score(predict_random, Y_test)
mat_RF = metrics.confusion_matrix(predict_random, Y_test)
tn_RF, fp_RF, fn_RF, tp_RF = mat_RF.ravel()
#_______________________________________________________________________________________________
from sklearn.neural_network import MLPClassifier
ann_classifier = MLPClassifier(hidden_layer_sizes=(100, ), max_iter=1000, activation='relu', random_state=42)
ann_classifier.fit(X_train, Y_train)
predict_ann = ann_classifier.predict(X_test) # Make predictions on the test data


#_____________________________________________________________________________________________
# Calculate performance metrics
r_ann = metrics.recall_score(predict_ann, Y_test)
p_ann = metrics.precision_score(predict_ann, Y_test)
acc_ann = metrics.accuracy_score(predict_ann, Y_test)
f1_ann = metrics.f1_score(predict_ann, Y_test)
mat_ann = metrics.confusion_matrix(predict_ann, Y_test)
tn_ann, fp_ann, fn_ann, tp_ann = mat_ann.ravel()
#_____________________________________________________________________________________________
dataa = [ ["Decision Trees", r, p, acc, f1,tn,fp,fn,tp], 
         ["Naive bayes", r_naive, p_naive, acc_naive, f1_naive,tn_naive,fp_naive,fn_naive,tp_naive],
         ["Random Forest", r_RF, p_RF, acc_RF, f1_RF,tn_RF,fp_RF,fn_RF,tp_RF],
        ["ANN", r_ann, p_ann, acc_ann, f1_ann, tn_ann, fp_ann, fn_ann, tp_ann],
       ] 
headers = ["Algorithm", "Recall", "Precision", "Accuracy", "F1 Score","TN","FP","FN","TP"]
table = tabulate(dataa, headers, tablefmt="pipe")
print(table)
max=0
if f1>max:
    max= f1
    best="Decision Trees algorithim"
if f1_naive>max: 
    max= f1_naive
    best="Naive bayes algorithim"

if f1_ann>max: 
    max= f1_ann
    best="ANN algorithim"

if f1_RF>max: 
    max= f1_RF
    best="Random Forest algorithim" 

print("\n\n the best algorithim depend on F1-score is--> ",best)           
#-----------------------------------------------------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk
from tabulate import tabulate

def ch():
    # Create the main application window
    app = tk.Tk()
    app.title("Algorithm Results")

    
    # Create a Button widget to trigger the action
    button = tk.Button(app, text="Print Table", command=lambda: (
    table_text.config(state=tk.NORMAL),
    table_text.delete(1.0, tk.END),
    table_text.insert(tk.END, tabulate(dataa, headers, tablefmt="pipe")),
    table_text.config(state=tk.DISABLED)
))
    button.pack(pady=10)
    # Create a Text widget to display the table
    table_text = tk.Text(app, height=10, width=100)
    table_text.pack(pady=10)
    table_text.config(state=tk.DISABLED)
    
    button_best_algorithm = tk.Button(app, text="Print Best Algorithm", command=lambda: best_label.config(
        text=f"\n The best algorithm, based on F1-score, is --> {best}"
    ))
    button_best_algorithm.pack(pady=10)

    # Create a Label widget to display the best algorithm
    best_label = tk.Label(app, text="")
    best_label.pack(pady=10)


# Start the main event loop
def button_click():
    gender_choice = gender_combobox.get()
    married_choice = married_combobox.get()
    dependents_choice = dependents_combobox.get()
    education_choice = education_combobox.get()
    self_employed_choice = self_employed_combobox.get()
    ApplicantIncome_choice = user_appl_income_entry.get()
    ApplicantIncome_choice = int(ApplicantIncome_choice) / 100
    Loan_Amount_choice=user_loanAmount_entry.get()
    Loan_Amount_Term_choice=user_Loan_Amount_Term_entry.get()
    History_choice=user_History_cb.get()
    Property_Area_choice=user_Property_Area_cb.get()
    '''
    print("Gender:", gender_choice)
    print("Marital Status:", married_choice)
    print("Dependents:", dependents_choice)
    print("Education:", education_choice)
    print("Self Employed:", self_employed_choice)
    print("Applicant Income:", ApplicantIncome_choice)
    print("Loan Amount:", Loan_Amount_choice)
    print("Loan Amount Term:", Loan_Amount_Term_choice)
    print("Credit History:", History_choice)
    print("Property Area:", Property_Area_choice)
    '''
    user_input = {
        "Gender": [0] if gender_choice == "Male" else [1],  # Assuming 1 for Male and 0 for Female
        "Married": [1] if married_choice == "Yes" else [0],  # Assuming 1 for Yes and 0 for No
        "Dependents": [3] if dependents_choice=="+3" else [int(dependents_choice)],
        "Education": [1] if education_choice == "Graduate" else [0],  # Assuming 1 for Graduate and 0 for Not Graduate
        "Self_Employed": [1] if self_employed_choice == "Yes" else [0],  # Assuming 1 for Yes and 0 for No
        "ApplicantIncome": [ApplicantIncome_choice],
        "LoanAmount": [Loan_Amount_choice],
        "Loan_Amount_Term": [Loan_Amount_Term_choice],
        "Credit_History": [int(History_choice)]
    }
    
# Now, user_input contains the label-encoded values
    user_input_df = pd.DataFrame(user_input)


    m=Decision_tree.predict(user_input_df)

    if m== 1:
        DT_label.config(text="Decision tree-->Loan Status: Yes")
    else:
        DT_label.config(text="Decision tree-->Loan Status: No")
    
    m1=naive_bayess.predict(user_input_df) #pridiction on test data (تطبيق)

    if m1== 1:
        N_bayes_Lable.config(text="Naive Bayes--->Loan Status: Yes")
    else:
        N_bayes_Lable.config(text="Naive Bayes--->Loan Status: No")

    ann_classifier.fit(X_train, Y_train)
    m2 = ann_classifier.predict(user_input_df)

    if m2== 1:
        Ann_Lable.config(text="Ann--->Loan Status: Yes")
    else:
        Ann_Lable.config(text="Ann--->Loan Status: No")

    random_foresrt.fit(X_train, Y_train)
    m3 = random_foresrt.predict(user_input_df)

    if m3== 1:
        RF_Lable.config(text="random_foresrt--->Loan Status: Yes")
    else:
        RF_Lable.config(text="random_foresrt--->Loan Status: No")


# Create window
app = tk.Tk()
app.title("Select Attributes")
app.geometry("800x800")  # Set desired width and height in pixels
from PIL import Image, ImageTk
image_path = "Ai.png"  #the path of image file
original_image = Image.open("Ai.png")
resized_image = original_image.resize((200, 200))
tk_image = ImageTk.PhotoImage(resized_image)

# Create a Label widget to display the image
image_label = tk.Label(app, image=tk_image)
image_label.pack(side=tk.LEFT, anchor=tk.NW, padx=10, pady=10)  # Adjust padx and pady

image_label.pack(pady=10)

    # Create a variable to store the selected gender
gender_var = tk.StringVar()
gender_combobox = ttk.Combobox(app, textvariable=gender_var, values=["Male", "Female"])
gender_combobox.set("Select Gender")

    # Create a variable to store the selected marital status
married_var = tk.StringVar()
married_combobox = ttk.Combobox(app, textvariable=married_var, values=["Yes", "No"])
married_combobox.set("Select Marital Status")

    # Create a variable to store the selected number of dependents
dependents_var = tk.StringVar()
dependents_combobox = ttk.Combobox(app, textvariable=dependents_var, values=["0", "1", "2", "+3"])
dependents_combobox.set("Select Dependents")

    # Create a variable to store the selected education level
education_var = tk.StringVar()
education_combobox = ttk.Combobox(app, textvariable=education_var, values=["Graduate", "Not Graduate"])
education_combobox.set("Select Education")

    # Create a variable to store the selected self-employed status
self_employed_var = tk.StringVar()
self_employed_combobox = ttk.Combobox(app, textvariable=self_employed_var, values=["Yes", "No"])
self_employed_combobox.set("Select Self Employed")

    # Create a variable to store the selected Income

user_appl_income = tk.StringVar()
user_appl_income_entry = ttk.Entry(app, textvariable=user_appl_income)
user_appl_income.set("ApplicantIncome")

        # Create a variable to store the selected loan amount

user_loanAmount = tk.StringVar()
user_loanAmount_entry = ttk.Entry(app, textvariable=user_loanAmount)
user_loanAmount.set("LoanAmount")

    # Create a variable to store the selected term

user_Loan_Amount_Term = tk.StringVar()
user_Loan_Amount_Term_entry = ttk.Entry(app, textvariable=user_Loan_Amount_Term)
user_Loan_Amount_Term.set("Loan_Amount_Term")  

    # Create a variable to store the selected History

user_History = tk.StringVar()
user_History_cb = ttk.Combobox(app, textvariable=user_History, values=["0","1"])
user_History_cb.set("Credit_History")

        # Create a variable to store the selected area

user_Property_Area = tk.StringVar()
user_Property_Area_cb = ttk.Combobox(app, textvariable=user_Property_Area, values=["Urban", "Rural", "Semiurban"])
user_Property_Area_cb.set("Property_Area")


    # Create a button
button = tk.Button(app, text="Check the predict", command=button_click)
    # Create a label to display the result
label = tk.Label(app, text="")

    # Create labels for results
DT_label = tk.Label(app, text="")
DT_label.pack(pady=10)
# Create a label to display the result
N_bayes_Lable = tk.Label(app, text="")
N_bayes_Lable.pack(pady=10)

Ann_Lable = tk.Label(app, text="")
Ann_Lable.pack(pady=10)

RF_Lable = tk.Label(app, text="")
RF_Lable.pack(pady=10)
    # Pack the comboboxes into the window
    # Pack the button into the window
    # Pack the  label into the window
gender_combobox.pack(pady=10)
married_combobox.pack(pady=10)
dependents_combobox.pack(pady=10)
education_combobox.pack(pady=10)
self_employed_combobox.pack(pady=10)
user_appl_income_entry.pack(pady=10)
user_loanAmount_entry.pack(pady=10)
user_Loan_Amount_Term_entry.pack(pady=10)
user_History_cb.pack(pady=10)
user_Property_Area_cb.pack(pady=10)
button.pack(pady=10)

label.pack()


button1 = tk.Button(app, text="Show the Alghorithim", command=ch)
button1.pack(pady=10)

app.mainloop()
