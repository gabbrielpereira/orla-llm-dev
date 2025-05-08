import os
import time
import chromadb
from firecrawl import FirecrawlApp
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
import dotenv

dotenv.load_dotenv()

#### SCRAPPING ####

#Credential
app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))

#Global variables

URL = 'https://www.orla.tech/'
SCRAPPING_DIR = 'scrape_output'
FILENAME = 'orla.md'
ORLA_VECTORDB = "orla_vectordb"

scrapping_dir_path = os.path.join(SCRAPPING_DIR, FILENAME)

# Map a website:
map_result = app.map_url(URL)
links_to_scrape = map_result.links

# Scrape each mapped link
for i in map_result.links:

    time.sleep(10) #wait before scrapping

    scrape_result = app.scrape_url(
        i, 
        formats=['markdown'], 
        only_main_content=True,
        )
    
    page_markdown_result = scrape_result.markdown #Page's markdown content

    #Append to markdown file
    try:
        with open(scrapping_dir_path,'a', encoding= "utf-8") as file:
            file.write(page_markdown_result)
        print(f"Markdown content of '{i}' added successfully.")
    except Exception as e:
        print(f"An error occurred while extracting the page {i}: {e}")


#### SPLITTING E EMBEDDINGS ####

markdown_content = page_markdown_result

headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

#Markdown document split
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
markdown_splits = markdown_splitter.split_text(markdown_content)
markdown_chunks=[]
for i in markdown_splits:
    markdown_chunks.append(i.page_content)

#Create ids for ChromaDB
ids_list = [str(i) for i in range(1, 59)]

#ChromaDB client and collection
persistent_client = chromadb.PersistentClient()
orla_collection = persistent_client.get_or_create_collection(ORLA_VECTORDB)

#Add chunks to collection
orla_collection.add(ids=ids_list, documents=markdown_chunks)
vector_store_from_client = Chroma(
    client=persistent_client,
    collection_name=ORLA_VECTORDB,
)
