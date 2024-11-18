from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain



# Load environment variables from the .env file
load_dotenv()

# Access the environment variables just like you would with os.environt
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")


# Response Format For my LLM Model
def LLMResponse(input_text,Number_of_Words,Style):

    # Define the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-002",temperature=0.2,api_key=GOOGLE_API_KEY)  

    # Prompt Template
    TEMPLATE = """
    You are an expert in generating detailed, coherent, and insightful responses tailored for various audiences and contexts. 
    Please generate a response in a **{Style} tone** on the topic: **"{input_text}"**. 

    Your answer should:
    - Stay within **{Number_of_Words} words**, ensuring conciseness and clarity.
    - Begin with a brief overview to establish context.
    - Include specific, relevant examples or analogies if they help explain complex points more effectively.
    - Address potential questions the reader might have to create a thorough, well-rounded response.

    Remember to avoid excessive jargon and aim for readability, especially if the chosen tone is "informal" or "student-friendly."
    """

    # Define prompt template and LLM chain
    prompt = PromptTemplate(input_variables=["Style", "input_text", "Number_of_Words"], template=TEMPLATE)
    llm_chain = LLMChain(llm=llm, prompt=prompt)

    # Generate response
    response = llm_chain.run({"Style": Style, "input_text": input_text, "Number_of_Words": Number_of_Words})
    print(response)
    return response




# Setting up the streamlit
st.set_page_config(page_title="Text Generation",
                   page_icon=":flag-tg:",
                   layout="centered",
                   initial_sidebar_state="collapsed")

# Heading
st.header("Text Generation")

# Input
input_text=st.text_input("Enter The Topic")

# Parameters
column_1,column_2=st.columns([5,5])

with column_1:
    Number_of_Words=st.text_input("Number Of Words")
with column_2:
    Style=st.selectbox('Tone of Writing', ["Professional Tone", "Informal Tone", "Students", "Normal"], index=0)



# Generate text on submit
if st.button("Generate Text"):
    if input_text and Number_of_Words.isdigit():
        response = LLMResponse(input_text, Number_of_Words, Style)
        st.write(response)
    else:
        st.error("Please enter a valid topic and number of words.")





