from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
from sklearn.naive_bayes import GaussianNB
import random
import string
from flask_cors import CORS
from sqlalchemy.exc import IntegrityError
from sqlalchemy import collate
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import requests
from flask import jsonify

import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

global done_list
done_list = []
global display_temp
display_temp = []
global final_list
global final_sym
final_sym = []
data = pd.DataFrame({"Symptom" : [] , "Severity" : [] , "Duration" : []})
train_medic_df = pd.read_csv('Merge.csv')
test_medic_df = pd.read_csv('LiverTest.csv')
train_final = train_medic_df.copy()
test_final = test_medic_df.copy()
train_final['prognosis'].nunique()

testmap = {
           'prognosis': {'AIDS' : 10, 'Acne' : 11,'Acute Liver Failure': 12, 'Alcoholic Hepatitis' : 13, 'Allergy' : 14, 'Arthritis' : 15,
                     'Autoimmune Hepatitis': 16, 'Bronchial Asthma' : 17, 'Cervical Spondylosis' : 18, 'Chicken Pox' : 19,
                     'Cholestasis' : 20, 'Common Cold' : 21, 'Covid' : 22, 'Dengue' : 23,
                     'Diabetes ' : 24, 'Piles' : 25, 'Adverse Drug Reaction' : 26, 'Liver Cirrhosis': 27,
                     'Fungal Infection' : 28, 'Acid Reflux' : 29, 'Gastroenteritis' : 30, 'Heart attack' : 31, 'Hemochromatosis': 32,
                     'Hepatitis A' : 33, 'Hepatitis B' : 34, 'Hepatitis C' : 35, 'Hepatitis D' : 36,
                     'Hepatitis E' : 37, 'Hypertension ' : 38, 'Hyperthyroidism' : 39, 'Hypoglycemia' : 40,
                     'Hypothyroidism' : 41, 'Impetigo' : 42 , 'Jaundice' : 43 , 'Malaria' : 44 , 'Migraine' : 45 ,
                     'Osteoarthritis' : 46 , 'Brain Bleed' : 47 ,'Vertigo' : 48,
                     'Peptic Ulcer Disease' : 49, 'Pneumonia' : 50, 'Psoriasis' : 51, 'Tuberculosis' : 52,
                     'Typhoid' : 53, 'Urinary Tract Infection' : 54, 'Varicose Veins' : 55, "Wilson's Disease": 56, 'Healthy': 57}
          }

map = {
       10 : ['Infectious Disease Specialist', 'Internist'], 11 : ['Dermatologist'], 12 : ['Gastroentologist','Internist'], 13 : ['Hepatologist', 'Gastroenterologist'],
       14 : ['Allergist'], 15 : ['Rheumatologist', 'Orthopedic surgeon'], 16 : ['Hepatologist'], 17 : ['Pulmonologist'], 18 : ['Orthopedic', 'Neurologist'],
       19 : ['Physician', 'Dermatologist'], 20 : ['Gastroenterologist', 'Hepatologist'], 21 : ['Physician'], 22 : ['Physician', 'Infectious disease specialist'],
       23 : ['Physician', 'Infectious disease specialist'], 24 : ['Diabetologist', 'Endocrinologist'], 25 : ['Proctologists'], 26 : ['Allergist', 'Pharmacologist'],
       27 : ['Hepatologist'], 28 : ['Dermatologist'], 29 : ['Gastroenterologist'], 30 : ['Gastroenterologist'], 31 : ['Cardiologist'],
       32 : ['Hematologist', 'Hepatologist'], 33 : ['Hepatologist', 'Gastroenterologist'],
       34 : ['Hepatologist', 'Gastroenterologist'], 35 : ['Hepatologist', 'Gastroenterologist'], 36 : ['Hepatologist', 'Gastroenterologist'],
       37 : ['Hepatologist', 'Gastroenterologist'], 38 : ['Cardiologist'], 39 : ['Endocrinologist'], 40 : ['Endocrinologist'],
       41 : ['Endocrinologist'], 42 : ['Dermatologist'], 43 : ['Gastroenterologist'], 44 : ['Infectious Disease Specialist'], 45 : ['Neurologist'],
       46 : ['Rheumatologist', 'Orthopedic surgeon'], 47 : ['Neurologist'],
       48 : ['Chiropractic Neurologist'], 49 : ['Gastroenterologist'],
       50 : ['Pulmonologist'], 51 : ['Dermatologist'], 52 : ['Infectious Disease Specialist'], 53 : ['Physician', 'Pediatrician'],
       54 : ['Gynecologist', 'Urologist'], 55 : ['Phlebologist', 'Vascular surgeon'], 56 : ['Hepatologist'], 57 : ['Physician']
      }


train_final = train_final.replace(testmap)
test_final = test_final.replace(testmap)

l = list(train_final.columns)
l.pop()
X = train_final[l]
y = train_final.prognosis
X_test = test_final[l]
y_test = test_final.prognosis

gnb = GaussianNB()
y_pred = gnb.fit(X, y).predict(X_test)
pred = gnb.fit(X, y).predict(X_test)
submit = pd.DataFrame(pred,index=X_test.index,columns=['Prognosis'])
s = train_medic_df.iloc[:1].copy()
s.iloc[0] = 0

final_sym= []
dis_list = []
doc_list = []

app = Flask(_name_)
CORS(app)
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+mysqlconnector://root:Anu12345!@localhost/usersdb'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


#a class named "Symptom" in a database model. 
# It contains various attributes, such as "sno" (serial number), "data_id" (data identifier), "name," "age," "gender," "smoking," "drinking," "symptom," "questions," "prognosis," and "doctors."
# These attributes are used to store information related to patients' symptoms, medical history, prognosis, and assigned doctors in a medical database.

class Symptom(db.Model):
    sno = db.Column(db.Integer, primary_key=True)
    data_id = db.Column(db.Integer, nullable=False)
    name = db.Column(db.String(200), nullable=False, default='')
    age = db.Column(db.Integer, nullable=False, default='')
    gender = db.Column(db.String(100), nullable=False, default='')
    smoking = db.Column(db.String(100), nullable=False, default='')
    drinking = db.Column(db.String(100), nullable=False, default='')
    symptom = db.Column(db.String(500), default='')
    questions = db.Column(db.String(10000), default='')
    prognosis = db.Column(db.String(200), default='')
    doctors = db.Column(db.String(200), default='')
    #date_created = db.Column(db.DateTime, default=datetime.utcnow)
    
    
#The provided API route '/analyze' is designed to receive POST requests and analyze symptom data. 
# It takes in a JSON payload containing patient information like name, age, gender, smoking and drinking habits, symptoms, prognosis, and doctors' details. 
# It then stores the data in the database and returns a success message. 
# Additionally, the API assumes that additional questions related to symptoms may be provided as an object and appends them to the 'questions' list.#

@app.route('/analyze', methods=['POST'])
def analyze():
    payload = request.json

    data_id = generate_data_id()
    name = payload.get('name')
    age = payload.get('age')
    gender = payload.get('gender')
    smoking = payload.get('smoking')
    drinking = payload.get('drinking')
    symptom = payload.get('symptom')
    prognosis = payload.get('prognosis')
    doctors = payload.get('doctors')
    
    questions = []

    # Assuming questions are provided as an object
    for key, value in payload.items():
        if key.startswith("question"):
            questions.append(value)

    questions_str = "/".join(questions)

    symptom= Symptom(
        data_id=data_id,
        name=name,
        age=age,
        gender=gender,
        smoking=smoking,
        drinking=drinking,
        symptom=symptom,
        prognosis=prognosis,
        doctors=doctors,
        questions=questions_str
    )

    db.session.add(symptom)
    db.session.commit()

    return 'Symptom data inserted successfully'



def generate_data_id():
    while True:
        # Generate a random 6-digit numeric user ID
        data_id = ''.join(random.choices(string.digits, k=6))

        # Check if the generated ID already exists in the database
        existing_symptom = Symptom.query.filter_by(data_id=data_id).first()
        if not existing_symptom:
            break

    return data_id

def get_data_id():
    # Retrieve the latest data_id from the database
    latest_symptom = Symptom.query.order_by(Symptom.sno.desc()).first()
    if latest_symptom:
        return latest_symptom.data_id
    else:
        return None

def _repr_(self):
        return f"Symptom(data_id={self.data_id}, questions={self.questions})"


#API route that handles a POST request to generate a medical recommendation based on symptoms provided in the payload.
# The code first checks if the 'symptom' parameter is present; otherwise, it returns an error message.
# It then generates a unique data ID for each recommendation request, filters dynamic questions based on the symptom and DataFrame, and finally calls the 'recommend' function to produce the response.

@app.route('/api', methods=['POST'])
def api():
    payload = request.json
    symptom = payload.get('symptom', None)

    if symptom is None:
        return jsonify({"error": "Invalid request. 'symptom' parameter is missing."})

    # Generate a new data ID for each recommendation request
    data_id = get_data_id() # Replace with your actual data ID generation logic

    # Get the dynamic questions based on the symptom and DataFrame
    display_temp = filter(symptom, train_medic_df)

    # Call the recommend function to ask questions and generate response
    return recommend(symptom, data_id)

#filters takes two parameters: "symptom" and "df." It operates on a pandas DataFrame, "qwerty," and aims to filter and extract relevant information based on the specified symptom. 
# The function creates "row_list" and "col_list" to gather rows and columns with the symptom value of 1.
# It then generates a "final_list" of symptoms sorted by their occurrence frequency. 
# The function ensures that all 12 questions are asked to the user, keeping track of previously asked questions in the "done_list." 
# Finally, it returns a list of "display_temp" containing questions that need to be presented to the user, filtered based on previous interactions.#


def filter(symptom, df):
    global done_list
    qwerty = df.copy()
    qwerty = qwerty.drop('prognosis', axis=1)

    row_list = []
    qwer = list(qwerty.columns)
    for it in range(len(qwerty.index)):
        val = qwerty.loc[[it], [symptom]].values
        if val == 1:
            row_list.append(it)

    short = qwerty.loc[row_list]
    short.reset_index(drop=True, inplace=True)

    col_list = []
    qwer = list(short.columns)
    for it in qwer:
        flag = 0
        for valu in short[it].values:
            if valu == 1:
                col_list.append(it)
                break

    a = {}
    for i in col_list:
        a[i] = 1

    final_list = []
    for key, value in a.items():
        final_list.append([key, short[key].value_counts()[1]])
    final_list.sort(key=lambda x: x[1], reverse=True)
    finalize = [item[0] for item in final_list]

    # Ensure all 12 questions are asked to the user
    display_temp = [q for q in finalize if q not in done_list]

    if len(display_temp) > 12:
        display_temp = display_temp[:12]

    done_list.extend(display_temp)

    return display_temp

#The function checks if the symptom is already present in a list called "final_sym." 
# If the symptom is present, it returns a JSON response with the "data_id" and no additional question to ask the user. 
# However, if the symptom is not present, it appends a question related to the symptom to a list called "response" and stores it in the database before returning the JSON response. 
# The question asks the user if they are experiencing the symptom provided as input.

def recommend(symptom, data_id):
    response = []

    # Check if the symptom is already present in the final_sym list
    if symptom in final_sym:
        # If the symptom is already present, do not ask the user if they are experiencing it
        return jsonify({"data_id": data_id, "question": None, "option": None})

    else:
        # If the symptom is not present, ask the user if they are experiencing it
        response.append({
            "data_id": data_id,
            "question": f"Are you experiencing {symptom}?",
            "option": "q0"  # Define the variable q0
        })
        # Store the question in the database
        store_question_in_db(data_id, response)

        return jsonify(response)
    
#stores generated questions in a database based on the data_id. 
# It first checks if a corresponding symptom entry exists in the database. 
# If so, it appends the new questions to the existing symptom's questions field.
# If not, it creates a new entry for the symptom and saves the questions in the questions field.
# The function utilizes the SQLAlchemy library to interact with the database.

def store_question_in_db(data_id, questions):
    # Store the generated questions in the database for the corresponding data_id
    existing_symptom = Symptom.query.filter_by(data_id=data_id).first()

    if existing_symptom:
        for question in questions:
            existing_symptom.questions += f"/{question['question']}"

        db.session.commit()
    else:
        # Create a new entry in the database for the symptom
        new_symptom = Symptom(data_id=data_id)
        for question in questions:
            new_symptom.questions = f"/{question['question']}"

        db.session.add(new_symptom)
        db.session.commit()
        
#"details(symptom)" that takes a symptom as input and collects information about its severity and duration. 
# It appends this data to the 'response' list and saves the collected information to a pandas DataFrame.
# The code also checks if all symptoms have been addressed and includes a 'Diagnosis completed' message accordingly. 
# The function then returns the 'response' list as a JSON object.

def details(symptom):
    response = []

    response.append("\nLet's talk about " + symptom)

    a = symptom

    b = request.json.get('severity')
    c = request.json.get('duration')
    q1 = "q1"  # Define the variable q1
    q2 = "q2"

    response.append({f"How severe is the {symptom}?": b, "option": q1})
    response.append({f"Since how long has it been affecting you?": c, "option": q2})

    # Assuming 'data' is a pandas DataFrame
    data.loc[len(data)] = [a, b, c]

    if len(final_sym) == len(l):  # Last symptom
        response.append({'message': 'Diagnosis completed'})  # Add the diagnosis completed message

    response.append('\n')

    return jsonify({'response': response})

#"diagnose." It appears to be a diagnostic system, likely for medical purposes. 
# The function uses the GaussianNB classifier to predict the prognosis based on the input data X and y, and then it displays the diagnosis results. 
# The top three probable diagnoses are stored in a list 'z,' 
# the function returns this list along with a response indicating the completion of the diagnosis process.

def diagnose():
    X = train_final[l]
    y = train_final.prognosis
    X_test = s[l]
    y_test = s.prognosis

    z = []

    gnb = GaussianNB()
    y_pred = gnb.fit(X, y).predict(X_test)

    probabilities = gnb.predict_proba(s[l])
    top_3_indices = (-probabilities).argsort()[:, :3]
    top3 = gnb.classes_[top_3_indices]

    z = top3[0].tolist()

    display()

    return jsonify({'z': z, 'response': 'Diagnosis completed'})

#display() that processes medical data and generates a response based on the input value 'z'. 
# It populates dis_list and doc_list with corresponding medical diagnoses and recommended doctors, respectively.
# The function also returns a customized response based on the diagnosis and suggests appropriate medical fields for consultation.#

def display():
    dis_list = []
    doc_list = []

    z = []
    # Loop through each value in 'z'
    for val in z:
        # Find the corresponding key in 'testmap'
        for key, value in testmap['prognosis'].items():
            if val == value:
                a = key
                dis_list.append(a)

        # Find the corresponding value in 'map'
        b = map.get(val, [])

        if val == 57:
            response = "There should be no major problem.\nHowever, you should consult a General Physician if the problem persists for more than 2 days."
        else:
            response = f"\nYou may have a case of {a}. You should consult a doctor in the following field/fields:\n"

            for ele in b:
                doc_list.append(ele)
                response += f"-->{ele}\n"

    # Assuming 's' is a DataFrame, set the value of the first row to 0
    s.iloc[0] = 0

    return jsonify({'dis_list': dis_list, 'doc_list': doc_list, 'response': response})


#The provided API, located at the route '/api/question', handles POST requests and processes user input related to symptoms. 
# It retrieves existing symptoms based on 'data_id', asks questions related to the symptom, and records user responses ('yes' or 'no').
# The API dynamically generates questions and stores them in the database.
# If the user response is 'yes', the API returns additional details about the symptom using the 'details' function.#


@app.route('/api/question', methods=['POST'])
def question():
    payload = request.json
    data_id = payload.get('data_id')
    user_input = payload.get('user_input')

    existing_symptom = Symptom.query.filter_by(data_id=data_id).first()

    if existing_symptom:
        questions = existing_symptom.questions.split('/')
        current_question = None

        if len(questions) > 0:
            current_question = questions[0]
            questions = questions[1:]  # Remove the first question as it's already asked

            # Append the user's response (yes/no) to the existing question
            if current_question:
                current_question += f" (User Response: {user_input})"
                questions.append(current_question)

            # Store the updated questions in the database
            existing_symptom.questions = '/'.join(questions)
            db.session.commit()

            if user_input.lower() == 'yes':
                final_sym.append(existing_symptom.symptom)
                return details(existing_symptom.symptom)  # Call the details function

        if len(questions) > 0:
            response = {
                "data_id": data_id,
                "question": current_question,
                "option": f"q{len(done_list)}"
            }
        else:
            response = {
                "data_id": data_id,
                "question": None,
                "option": None
            }

        return jsonify(response)

    return jsonify({'error': 'Invalid data_id or no questions to ask.'})

#This API, located at '/api/generate_report', is designed to generate a detailed report based on a given 'data_id'.
# It retrieves symptom details from a database using the provided 'data_id', and if the data is found.
# it returns the relevant information, including name, age, gender, smoking and drinking habits, symptoms, questions asked, prognosis, and recommended doctors. 
# The response is provided in JSON format, along with a success message, confirming the successful receipt of the details.#

@app.route('/api/generate_report', methods=['POST'])
def generate_report():
    payload = request.json
    data_id = payload.get('data_id')

    # Retrieve symptom details based on data_id from the database
    symptom_details = Symptom.query.filter_by(data_id=data_id).first()

    if not symptom_details:
        return jsonify({"error": "Data not found for the given data_id."})

    name = symptom_details.name
    age = symptom_details.age
    gender = symptom_details.gender
    smoking = symptom_details.smoking
    drinking = symptom_details.drinking
    symptom = symptom_details.symptom
    questions = symptom_details.questions
    prognosis = symptom_details.prognosis
    doctors = symptom_details.doctors

    response = {
        "data_id": data_id,
        "name": name,
        "age": age,
        "gender": gender,
        "smoking": smoking,
        "drinking": drinking,
        "symptom": symptom,
        "questions": questions,
        "prognosis": prognosis,
        "doctors": doctors
    }

    return jsonify(response, "Details received successfully!")