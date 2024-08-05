from typing import List

import streamlit as st
import pandas as pd
import numpy as np
import pickle

from pandas import DataFrame

st.write("l'application qui prédit du crédit")

#Collecter le profil d'entrée
st.sidebar.header("Les Caractéristiques du clients")

def client_create_entree():
    Gender=st.sidebar.selectbox('Sex',('Male','Female'))
    Married=st.sidebar.selectbox('Marié',('Yes','No'))
    Dependents=st.sidebar.selectbox('Enfant',('0','1','2','3+'))
    Education=st.sidebar.selectbox('Education',('GRaduate','Not Graduate'))
    Self_Employed=st.sidebar.selectbox('Salirié ou Entrepreneur',('Yes','No'))
    ApplicantIncome=st.sidebar.slider('Salaire du client',150,4000,200)
    CoapplicantIncome=st.sidebar.slider('Salaire du conjoint',0,40000,2000)
    LoanAmount=st.sidebar.slider('Montant du crédit en Kdollar',9.0,700.0,200.0)
    Loan_Amount_Term=st.sidebar.selectbox('Durée du crédit',(360.0,120.0,240.0,180.0,60.0,300.0,36.0,84.0,12.0))
    Credit_History=st.sidebar.selectbox('Credit_History',(1.0,0.0))
    Property_Area=st.sidebar.selectbox('Property_Area',('Urban','Rural','Semiurban'))

    data={
        'Gender':Gender,
        'Married':Married,
        'Dependents':Dependents,
        'Education':Education,
        'Self_Employed':Self_Employed,
        'ApplicantIncome':ApplicantIncome,
        'CoapplicantIncome':CoapplicantIncome,
        'LoanAmount':LoanAmount,
        'Loan_Amount_Term':Loan_Amount_Term,
        'Credit_History':Credit_History,
        'Property_Area':Property_Area
    }

    profil_client=pd.DataFrame(data,index=[0])
    return profil_client

input_df=client_create_entree()


#Transformer les données d'entrées en données adaptées à notre modèle
#Importantion des donnés
df=pd.read_csv('train.csv')
credit_input=df.drop(columns=['Loan_ID','Loan_Status'])
donnee_entree=pd.concat([input_df,credit_input],axis=0)

#Encodage des données
donnee_entree['Gender'] = donnee_entree['Gender'].replace({'Male': 1, 'Female': 0})
donnee_entree['Married'] = donnee_entree['Married'].replace({'Yes': 1, 'No': 0})
donnee_entree['Dependents'] = donnee_entree['Dependents'].replace({'0': 0, '1': 1, '2': 2, '3+': 3})
donnee_entree['Education'] = donnee_entree['Education'].replace({'GRaduate': 1,'Graduate': 1, 'Not Graduate': 0})
donnee_entree['Self_Employed'] = donnee_entree['Self_Employed'].replace({'Yes': 1, 'No': 0})
donnee_entree['Credit_History'] = donnee_entree['Credit_History'].replace({'1.0': 1, '0.0': 0})
donnee_entree['Property_Area'] = donnee_entree['Property_Area'].replace({'Urban': 0,'Rural': 1,'Semiurban': 2})
#var_cat=['Gender', 'Married', 'Dependents', 'Education','Self_Employed','Credit_History', 'Property_Area']
#for col in var_cat:
    #dummy=pd.get_dummies(donnee_entree[col],drop_first=True)
    #donnee_entree=pd.concat([dummy,donnee_entree],axis=1)
    #del donnee_entree[col]
#Prendre uniquement la première ligne
donnee_entree=donnee_entree[:1]

#Afficher les données transformées
st.subheader('Les caractéristiques transformées')
st.write(donnee_entree)

#Importer le modèle
load_model=pickle.load(open('Prévision_crédit.pkl','rb'))


#Allpiquer le modèle sur le profil d'entrée
prevision=load_model.predict(donnee_entree)

st.subheader('Résultat de la prévision')
st.write(prevision)