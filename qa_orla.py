import os
import chromadb
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

CHROMA_PATH = "orla_vectordb"

PROMPT_TEMPLATE = """
    ####
    Você é Mar, uma assistente virtual de uma empresa de consultoria de TI chamada Orla, baseada em São Paulo-SP.
    Você é amigável, carísmática e solícita e sua resposta deve ser completa, concisa e refletir essas características.
    
    ####
    Sua tarefa é responder aos usuários perguntas sobre a empresa priorizando informações que foram extraídas do\
    site institucional e que estarão no seu contexto:
    {context}

    ####
    Você não deve:

    ## Responder perguntas ou especular sobre concorrentes diretos da Orla.
    ## Responder perguntas antiéticas ou discriminatórias.
    ## Inventar informações sobre a Orla ou sobre qualquer outro tema
    ## Responder perguntas fora do escopo da Orla não deve ser respondida.
    ## Se apresentar quando não for perguntada

    #### 
    Responda a pergunta {question} com o máximo de detalhes"""

### Models ###
embeddings_model = OllamaEmbeddings(model="llama3")

chat_model = ChatGroq(
    temperature=0.3,
    model="llama3-70b-8192",
    api_key=os.getenv("GROQ_API_KEY")
)


## ChromaDB client and collection ##
persistent_client = chromadb.PersistentClient()
orla_collection = persistent_client.get_collection("orla_vectordb")

def query_rag(question):
  
  # Retrieve results from query
  results = orla_collection.query(
    query_texts=[question], 
    )

  # Build prompt template
  prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
  prompt = prompt_template.format(context=results, question=question)
  
  # Generate response text based on the prompt
  response_text = chat_model.invoke(prompt)
 
  return response_text

#Faça sua pergunta
question = input("Por favor, digite sua pergunta. Caso não tenha mais perguntas, digite 'sair'.'\n")

while question != 'sair':
    response_text = query_rag(question)
    print('\n',response_text.content, '\n')
    question = input("Faça mais uma pergunta. Caso não tenha mais perguntas, digite 'sair'.'\n")