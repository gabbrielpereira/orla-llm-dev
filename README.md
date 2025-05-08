# Desafio Técnico – Chatbot Mar

Mar é uma assistente virtual da Orla Tech capaz de responder perguntas e fornecer informações institucionais detalhadas da Orla.

## Stack:
- Python 3.11
- Groq API 
- Firecrawl API
- Langchain
- ChromaDB

## Ambiente

Você deve: 

### 1.
Criar as chaves das API do Groq (https://groq.com/), para utilizar um modelo de IA Generativa, e da API do Firecrawl (https://www.firecrawl.dev/), para fazer crawling do site da orla (orla.tech).

### 2.
Adicionar as chaves das APIs no .env

### 3.
Instalar as dependências do requirements.txt

## Rodando o projeto

### Scrapping e embeddings do site

Na linha de comando, rode o arquivo scrapping_and_embeddings.py para realizar o scrappig do website, gerar um arquivo Markdown, e realizar o embedding do arquivo gerado no banco de dados vetorial do ChromaDB.

### Chatbot Mar

Para conversar com a Mar, você deve executar o arquivo qa_orla.py e na linha de comando fazer perguntas a ela como input.

### Validação das respostas

Para validar as respostas geradas pela Mar, execute o arquivo answer_evaluation.py. Ele possui um dataset de cinco perguntas e cinco respostas que podem ser editadas para outros datasets.


