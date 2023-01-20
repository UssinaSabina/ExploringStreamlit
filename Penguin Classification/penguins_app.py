import pandas as pd
import streamlit as st
import pickle

st.write('''
# Penguin Prediction App
This app predicts the **Palmer Penguin** species!
Data obtained from the [palmerpenguins library](https://github.com/allisonhorst/palmerpenguins) in R by Allison Horst.
''')

st.sidebar.header("User Input Features")
st.sidebar.markdown('''
[Example CSV input file](https://github.com/UssinaSabina/ExploringStreamlit/blob/main/Penguin%20Classification/penguins_example.csv)
''')

# creates dataframe from user input features
uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        island = st.sidebar.selectbox('Island', ('Biscoe', 'Dream', 'Torgersen'))
        sex = st.sidebar.selectbox('Sex', ('Male', 'Female'))
        bill_length_mm = st.sidebar.slider('Bill length (mm)', 32.1, 59.6, 43.9)
        bill_depth_mm = st.sidebar.slider('Bill depth (mm)', 13.1, 21.5, 17.2)
        flipper_length_mm = st.sidebar.slider('Flipper length (mm)', 172.0, 231.0, 201.0)
        body_mass_g = st.sidebar.slider('Body mass (g)', 2700.0,6300.0,4207.0)
        data = {
            'island': island,
            'bill_length_mm': bill_length_mm,
            'bill_depth_mm': bill_depth_mm,
            'flipper_length_mm': flipper_length_mm,
            'body_mass_g': body_mass_g,
            'sex': sex
        }
        input_features = pd.DataFrame(data, index=[0])
        return input_features
    input_df = user_input_features()

# combine entire dataset with user input features (it's useful for encoding)
penguins_raw = pd.read_csv('penguins_cleaned.csv')
# drop column with target values
penguins_raw.drop(columns=['species'], inplace=True)
df = pd.concat([input_df, penguins_raw], axis=0)
# ordinal feature encoding
encode = ['sex', 'island']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]
df =df[:1] # only first row (the user input data)


# display the user input features
st.subheader("User input features")
if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below)')
    st.write(df)


# read in saved classification model
load_clf = pickle.load(open('penguins_clf.pkl', 'rb'))

# make prediction
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

# display prediction
st.subheader('Prediction')
penguins_species = ['Adelie', 'Chinstrap', 'Gentoo']
st.write(penguins_species[prediction])
# display probabilities
st.subheader('Prediction Probability')
st.write(prediction_proba)
