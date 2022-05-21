### Imports [Please have these libraries installed wherever you are using in your flutter/flask environment]

import numpy as np
import random
from flask import Flask
import zipfile
from flask import jsonify
from flask import request
# Critical Imports for running the model 

from tensorflow import keras
import tensorflow as tf

#you are required to Unzip the bayesian_nn.zip & put the path of the folder here (folder name is bayesian_nn)


with zipfile.ZipFile('bayesian_nn.zip', 'r') as zip_ref:
    zip_ref.extractall('bayesian_nn')

model = keras.models.load_model('bayesian_nn')

# hardcoding some inputs: please modify the variable to take values from the flutter app
'''
gender = 'Male'
marital_status = 'Married'
age = '25-40 Years'
occupation = 'Corporate Employee'
income = '10-20 lac'
ailment = 'No'
'''

# Please use following input 

### Input Keys: {input_field:actual_value}
#### Gender: {0:female,1:male}
#### Marital Status: {0:Married,1:Unmarried}
#### Age: {0:25-40,1:41-60,2:Above 60, 3:Below 25}
#### Occupation: {0: Corporate Employee, 1: Enterpreneur, 2: Medical Professional, 3: Military Service, 4: Self Employed}
#### Income: {0: 0-5 lac,1: 10-20 lac,2: 20-30 lac,3: 30-40 lac,4: 40-50 lac,5: 5-10 lac}

#### Any Serious Medical Ailment: {0: No, 1: Yes}



def take_inputs(gender, marital_status,age, occupation, income, ailment):
  #Ask Questions:
  #global gender,marital_status,age,occupation, income,ailment
  #Commenting out lines not needed


  '''
  gender = input('What is your Gender {Male/Female}\n')
  marital_status = input('What is your Marital Status {Married/Unmarried}\n')
  age = input('What is your Age {Below 25 Years, 25-40 Years, 41-60 Years, Above 60 Years\n')
  occupation = input(' What is your occupation{Corporate Employee, Enterpreneur, Medical Professional, Military Service, Self Employed}\n')
  income = input('What is your income {0-5 lac,10-20 lac,20-30 lac,30-40 lac,40-50 lac,5-10 lac}\n')
  ailment = input('Did you have any Serious Medical ailment in past 2 years? {Yes/No}\n')
  '''
  #encode inputs 

  if(gender == 'Male'):
    gender = 0
  elif(gender == 'Female'):
    gender = 1
  
  if(marital_status == 'Married'):
    marital_status = 0
  elif(marital_status == 'Unmarried'):
    marital_status = 1

  if(age == 'Below_25_Years'):
    age = 3
  elif(age == '25_40_Years'):
    age = 0
  elif(age == '41_60_Years'):
    age = 1
  elif(age == 'Above_60_Years'):
    age = 2
  
  if(occupation == 'Corporate_Employee'):
    occupation = 0
  elif(occupation == 'Enterpreneur'):
    occupation = 1
  elif(occupation == 'Medical_Professional'):
    occupation = 2
  elif(occupation == 'Military_Service'):
    occupation = 3
  elif(occupation == 'Self_Employed'):
    occupation = 4

  if(income == '0_5_lac'):
    income = 0
  elif(income == '5_10_lac'):
    income = 5
  elif(income == '10_20_lac'):
    income = 1
  elif(income == '20_30_lac'):
    income = 2
  elif(income == '30_40_lac'):
    income = 3
  elif(income == '40_50_lac'):
    income = 4

  if(ailment == 'Yes'):
    ailment = 1
  elif(ailment == 'No'):
    ailment = 0

  encoded_inputs = [gender,marital_status,age,occupation,income]
  

  return encoded_inputs,ailment


def calculate_premium(gender, marital_status,age, occupation, income, ailment):
  policy_amt1 = np.array([800000,1000000,200000,600000,450000])
  policy_amt2 = np.array([900000,400000,150000,750000,350000])
  policy_amt3 = np.array([1200000,1400000,550000,1150000,650000])
  policy_amt4 = np.array([1500000,2000000,100000,950000,375000])
  policy_coverage_1 = np.array([600000,750000,160000,450000,350000])
  policy_coverage_2 = np.array([675000,300000,125000,575000,275000])
  policy_coverage_3 = np.array([900000,1100000,425000,875000,475000])
  policy_coverage_4 = np.array([1125000,1500000,75000,750000,275000])
  premium_1 = np.array([25000,25000,20000,40000,18000])
  premium_2 = np.array([30000,40000,10000,35000,25000])
  premium_3 = np.array([30000, 40000,25000,45000,30000])
  premium_4 = np.array([75000,80000,10000,45000,25000])
  tenure_1 = np.zeros_like(premium_1)
  tenure_2 = np.zeros_like(premium_2)
  tenure_3 = np.zeros_like(premium_3)
  tenure_4 = np.zeros_like(premium_4)
  inputs, health = take_inputs(gender, marital_status,age, occupation, income, ailment)
  t1 = 1000
  t2 = 10000
  if(health == 1):
    premium_1 = np.ceil(premium_1*1.3/t1) * t1
    premium_2 = np.ceil(premium_2*1.3/t1) * t1
    premium_3 = np.ceil(premium_3*1.3/t1) * t1
    premium_4 = np.ceil(premium_4*1.3/t1) * t1
    policy_coverage_1 = np.ceil(policy_coverage_1/t2) * t2
    policy_coverage_2 = np.ceil(policy_coverage_2/t2) * t2
    policy_coverage_3 = np.ceil(policy_coverage_3/t2) * t2
    policy_coverage_4 = np.ceil(policy_coverage_4/t2) * t2

  if(inputs[2] == 3):
    premium_1 = np.ceil(premium_1*0.75/t1) * t1
    premium_2 = np.ceil(premium_2*0.75/t1) * t1
    premium_3 = np.ceil(premium_3*0.75/t1) * t1
    premium_4 = np.ceil(premium_4*0.75/t1) * t1
    policy_coverage_1 = np.ceil(policy_coverage_1 * 1.1/t2) * t2
    policy_coverage_2 = np.ceil(policy_coverage_2 * 1.1/t2) * t2
    policy_coverage_3 = np.ceil(policy_coverage_3 * 1.1/t2) * t2
    policy_coverage_4 = np.ceil(policy_coverage_4 * 1.1/t2) * t2
  elif(inputs[2] == 0):
    premium_1 = np.ceil(premium_1*1.0/t1) * t1
    premium_2 = np.ceil(premium_2*1.0/t1) * t1
    premium_3 = np.ceil(premium_3*1.0/t1) * t1
    premium_4 = np.ceil(premium_4*1.0/t1) * t1
    policy_coverage_1 = np.ceil(policy_coverage_1*1.0/t2) * t2
    policy_coverage_2 = np.ceil(policy_coverage_2*1.0/t2) * t2
    policy_coverage_3 = np.ceil(policy_coverage_3*1.0/t2) * t2
    policy_coverage_4 = np.ceil(policy_coverage_4*1.0/t2) * t2
  elif(inputs[2] == 1):
    premium_1 = np.ceil(premium_1*1.2/t1) * t1
    premium_2 = np.ceil(premium_2*1.2/t1) * t1
    premium_3 = np.ceil(premium_3*1.2/t1) * t1
    premium_4 = np.ceil(premium_4*1.2/t1) * t1
    policy_coverage_1 = np.ceil(policy_coverage_1*0.9/t2) * t2
    policy_coverage_2 = np.ceil(policy_coverage_2*0.9/t2) * t2
    policy_coverage_3 = np.ceil(policy_coverage_3*0.9/t2) * t2
    policy_coverage_4 = np.ceil(policy_coverage_4*0.9/t2) * t2
  elif(inputs[2] == 2):
    premium_1 = np.ceil(premium_1*1.4/t1) * t1
    premium_2 = np.ceil(premium_2*1.4/t1) * t1
    premium_3 = np.ceil(premium_3*1.4/t1) * t1
    premium_4 = np.ceil(premium_4*1.4/t1) * t1
    policy_coverage_1 = np.ceil(policy_coverage_1*0.75/t2) * t2
    policy_coverage_2 = np.ceil(policy_coverage_2*0.75/t2) * t2
    policy_coverage_3 = np.ceil(policy_coverage_3*0.75/t2) * t2
    policy_coverage_4 = np.ceil(policy_coverage_4*0.75/t2) * t2
  if(inputs[0] == 0):
    premium_1 = np.ceil(premium_1*1.1/t1) * t1
    premium_2 = np.ceil(premium_2*1.1/t1) * t1
    premium_3 = np.ceil(premium_3*1.1/t1) * t1
    premium_4 = np.ceil(premium_4*1.1/t1) * t1
  elif(inputs[0] == 1):
    premium_1 = np.ceil(premium_1*0.9/t1) * t1
    premium_2 = np.ceil(premium_2*0.9/t1) * t1
    premium_3 = np.ceil(premium_3*0.9/t1) * t1
    premium_4 = np.ceil(premium_4*0.9/t1) * t1
  tenure_1 = np.ceil(policy_amt1/premium_1)
  tenure_2 = np.ceil(policy_amt2/premium_2)
  tenure_3 = np.ceil(policy_amt3/premium_3)
  tenure_4 = np.ceil(policy_amt4/premium_4)

  return inputs, health, policy_amt1,policy_amt2,policy_amt3,policy_amt4, policy_coverage_1,policy_coverage_2,policy_coverage_3,policy_coverage_4,premium_1,premium_2,premium_3,premium_4,tenure_1,tenure_2,tenure_3,tenure_4

def bayesian_neural_network(inputs):
  n_epochs = 25
  X_test = np.array([inputs, inputs])
  y_test = np.array([inputs, inputs])
  test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  test_set = test_set.shuffle(len(X_test), reshuffle_each_iteration=True)
  testing_data = test_set.take(len(X_test)).batch(1).repeat(n_epochs)
  samples = 2
  iterations = 10 
  test_iterator = tf.compat.v1.data.make_one_shot_iterator(testing_data)
  X_true, Y_true, Y_pred = np.empty(shape=(samples, len(inputs))), np.empty(shape=(samples, len(inputs))), np.empty(shape=(samples, len(inputs), iterations))
  for i in range(samples):
    features, labels = test_iterator.get_next()
    X_true[i,:] = features
    Y_true[i,:] = labels.numpy()
    for k in range(iterations):
        Y_pred[i,:,k] = model.predict(features)
    Y_pred_m = np.mean(Y_pred, axis=-1)
    y_pred = np.round(Y_pred_m)
    return y_pred[0]


def recommend_policy(gender, marital_status,age, occupation, income, ailment):
  inputs, health, policy_amt1,policy_amt2,policy_amt3,policy_amt4, policy_coverage_1,policy_coverage_2,policy_coverage_3,policy_coverage_4,premium_1,premium_2,premium_3,premium_4,tenure_1,tenure_2,tenure_3,tenure_4= calculate_premium(gender, marital_status,age, occupation, income, ailment)
  policy_recommendations = bayesian_neural_network(inputs)
  #print('Recommending Following Policies based on your preferences:')
  policy_number = []
  predicted_policy = []
  predicted_coverage = []
  predicted_premium = []
  predicted_tenure = []
  for i in range(0,5):
    if(policy_recommendations[i] == 1):
      if(inputs[2] == 1 or inputs[0] == 1):
        j = i + 5
        policy_number.append(i)
        predicted_policy.append(policy_amt1[i])
        predicted_coverage.append(policy_coverage_1[i])
        predicted_premium.append(premium_1[i])
        predicted_tenure.append(tenure_1[i])

        policy_number.append(j)
        predicted_policy.append(policy_amt2[i])
        predicted_coverage.append(policy_coverage_2[i])
        predicted_premium.append(premium_2[i])
        predicted_tenure.append(tenure_2[i])
        
      elif(inputs[2] == 3):
        k = i + 10
        l = i + 15
        policy_number.append(k)
        predicted_policy.append(policy_amt3[i])
        predicted_coverage.append(policy_coverage_3[i])
        predicted_premium.append(premium_3[i])
        predicted_tenure.append(tenure_3[i])
        
        policy_number.append(k)
        predicted_policy.append(policy_amt4[i])
        predicted_coverage.append(policy_coverage_4[i])
        predicted_premium.append(premium_4[i])
        predicted_tenure.append(tenure_4[i])

      elif(inputs[2] == 2):
        j = i + 5
        policy_number.append(i)
        predicted_policy.append(policy_amt2[i])
        predicted_coverage.append(policy_coverage_2[i])
        predicted_premium.append(premium_2[i])
        predicted_tenure.append(tenure_2[i])
        
  if(inputs[2] == 2 and inputs[1] == 0):
    policy_number.append(9)
    predicted_policy.append(policy_amt2[4])
    predicted_coverage.append(policy_coverage_2[4])
    predicted_premium.append(premium_2[4])
    predicted_tenure.append(tenure_2[4])
    
    policy_number.append(19)
    predicted_policy.append(policy_amt4[4])
    predicted_coverage.append(policy_coverage_4[4])
    predicted_premium.append(premium_4[4])
    predicted_tenure.append(tenure_4[4])
    
  if(all(policy_recommendations) == 0):
    ch = [2,3,4]
    select = random.choice(ch)
    policy_number.append(select)
    predicted_policy.append(policy_amt1[select])
    predicted_coverage.append(policy_coverage_1[select])
    predicted_premium.append(premium_1[select])
    predicted_tenure.append(tenure_1[select])
    
    select = random.choice(ch)
    policy_number.append(select+5)
    predicted_policy.append(policy_amt2[select])
    predicted_coverage.append(policy_coverage_2[select])
    predicted_premium.append(premium_2[select])
    predicted_tenure.append(tenure_2[select])
    

    select = random.choice(ch)
    policy_number.append(select+10)
    predicted_policy.append(policy_amt3[select])
    predicted_coverage.append(policy_coverage_3[select])
    predicted_premium.append(premium_3[select])
    predicted_tenure.append(tenure_3[select])
  return policy_number,predicted_policy,predicted_coverage,predicted_premium,predicted_tenure
  
  # ======================================================================4
  
  ## Flask Code 

  
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello Heroku Server! InsureBuddy Here!!!"

@app.route("/getpolicy", methods = ['GET','POST'])
def get_recommendation():
  gender = str(request.args.get('gender'))
  marital_status = str(request.args.get('marital_status'))
  age = str(request.args.get('age'))
  occupation = str(request.args.get('occupation'))
  income = str(request.args.get('income'))
  ailment = str(request.args.get('ailment'))
  policy_number,predicted_policy,predicted_coverage,predicted_premium,predicted_tenure = recommend_policy(gender, marital_status,age, occupation, income, ailment)
  policy_number = ['Policy '+ str(i) for i in policy_number]
  predicted_policy = ["Rs." + str(i)  for i in predicted_policy]
  predicted_premium = ["Rs." + str(i) for i in predicted_premium]
  predicted_coverage = ["Rs." + str(i) + ' + Additional' for i in predicted_coverage]
  predicted_tenure = [str(i) + ' Years' for i in predicted_tenure]
    
  output = ''
  output = output + '['
  for i in range(0,len(policy_number)):
    output = output + '{'
    output = output + '"policyAmount":"{}",'.format(predicted_policy[i])
    output = output + '"policyCoverage":"{}",'.format(predicted_coverage[i])
    output = output + '"policyNumber":"{}",'.format(policy_number[i])
    output = output + '"policyPremium":"{}",'.format(predicted_premium[i])
    output = output + '"policyTenure":"{}"'.format(predicted_tenure[i])
    if(i == len(policy_number) - 1):
     output = output + '}'
    else:
      output = output + '},'
  output = output + ']'

  return output


if __name__ == '__main__':
    app.run(debug=True)
