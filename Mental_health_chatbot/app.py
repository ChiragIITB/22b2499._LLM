import streamlit as st
from langchain.prompts import PromptTemplate 
from langchain.llms import CTransformers


## Function to get the response from the fine-tuned model

def getLLamaresponse(input_text):

    ### fine tuned - LLama 2 model

    llm = CTransformers(model='models/llama-2-7b-chat.ggmlv3.q8_0.bin', ## ENTER MODEL ADDRESS
                        model_type='llama',
                        config={'max_new_tokens':256,
                                'temperature':0.01}) 
    
        ## Prompt Template

    template="""
    You are professional therapist counselling a patient. A patient dealing with mental health trouble will be talking to you. I want you to console them and provide them with relevant suggestions to work on themselves.

    {input_text}

    Help the patient by giving him suggestions to work on the things he is going through. 
    """ 


    prompt = PromptTemplate(input_variables=["input_text"],
                            template =template)
    
    ## Generate the responses from the model 

    response = llm(prompt.format(input_text=input_text))
    print(response)
    return response


st.set_page_config(page_title="Mental Health Chatbot",
                   page_icon='✌️',
                   layout='centered', 
                   initial_sidebar_state='collapsed')

st.header("Mental Health Chatbot ✌️")

input_text = st.text_input("Heyy, how are you doing today ?")

## creating a submit button

submit = st.button("ask the bot")

## Final Response

if submit: 
    st.write(getLLamaresponse(input_text))