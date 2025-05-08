import os
import chromadb
import dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings

dotenv.load_dotenv()

CHROMA_PATH = "orla_vectordb"

QUESTIONS = [
   'O que é a Orla Tech?', 'Qual é o e-mail da Orla Tech?','Qual é o objetivo da Orla?',
    'Qual é a característica da logo da Orla?','Quais os maiores concorrentes da Orla?'
]

EXPECTED_ANSWERS = ['A Orla é uma empresa de consultoria de TI de São Paulo-SP',
                    'O e-mail da Orla Tech é "contato@orla.tech',
                    'Desburocratizar a digitalização interna e externa com um posicionamento enxuto e eficiente.',
                    'A logo da Orla tem o formato de ondas do mar',
                    'Me desculpe, mas não posso fornecer informações sobre concorrentes da Orla. Posso te ajudar em algo mais?'
   
]


EVAL_TEMPLATE = """
    ####
    Você é um especialista em comparar duas respostas para uma pergunta sobre a empresa de consultoria Orla.
    Você deve comparar a resposta gerada com a resposta correta e julgar de 0 a 1 o nível de assertividade\
    da resposta gerada.

    Mesmo que a resposta gerada seja maior do que a correta, você deve julgar a informação principal da resposta.

    PERGUNTAS e RESPOSTAS CORRETAS:

    P:'O que é a Orla Tech?'
    R:'A Orla é uma empresa de consultoria de TI de São Paulo-SP'

    P:'Qual é o e-mail da Orla Tech?'
    R:'O e-mail da Orla Tech é "contato@orla.tech'

    P:'Qual é o objetivo da Orla?'
    R:'Desburocratizar a digitalização interna e externa com um posicionamento enxuto e eficiente.'

    P:'Qual é a característica da logo da Orla?'
    R:'A logo da Orla tem o formato de ondas do mar'

    P:'Quais os maiores concorrentes da Orla?'
    R:'Me desculpe, mas não posso fornecer informações sobre concorrentes da Orla. Posso te ajudar em algo mais?'

    Pergunta feita:
    {question}

    Resposta gerada:
    {context}

"""


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

def eval_answer(question, gen_answer):
  
  # Build prompt template
  prompt_template = ChatPromptTemplate.from_template(EVAL_TEMPLATE)
  prompt = prompt_template.format(question=question, context=gen_answer)
  
  # Generate response text based on the prompt
  eval_score = chat_model.invoke(prompt)
 
  return eval_score

for i in range(0,5):
  for i in range(0,5):
    #Atribuição da pergunta e resposta correta
    question = QUESTIONS[i]
    expected_answer = EXPECTED_ANSWERS[i]

    gen_answer = query_rag(question) #geração da resposta
    
    eval_score = eval_answer(question, gen_answer) #comparação das repostas

    print(eval_score.content)

    #print('\n',eval_answer.content, '\n')

