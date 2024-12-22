from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import SentenceTransformersTokenTextSplitter
from typing import Optional
from .custom_embeddings import CustomEmbeddings


class SemanticChunkerConfig:
    @classmethod
    def create(cls, embeddings_path: str = "jinaai/jina-embeddings-v2-base-en",
                device: Optional[str] = "mps", 
                breakpoint_threshold_type: str = "percentile"):
        embeddings = CustomEmbeddings(embeddings_path, device)

        return SemanticChunker(embeddings, 
                               breakpoint_threshold_type=breakpoint_threshold_type
        )


class TokenChunkerConfig:

    @classmethod
    def create(cls, chunk_overlap: int = 30, tokens_per_chunk: int = 200):

        return SentenceTransformersTokenTextSplitter(chunk_overlap=chunk_overlap, model_name='sentence-transformers/all-mpnet-base-v2', tokens_per_chunk=tokens_per_chunk)





def chunk_data(df, config):
    """This function is supposed to chunk data into smaller text blocks

    @param: df represents the dataframe that contains the data/documents to be chunked
            in the column "cleaned_content"
    @return: list of all chunks created in order
    @return: dict where the key is the index of the chunk in the list, and the value is the chunk itself
    """


    # text_splitter = SemanticChunkerConfig.create(config["Paths"]["embeddings_path"], config["General"]["device"])
    text_splitter = TokenChunkerConfig.create(config["Chunking"]["chunk_overlap"], config["Chunking"]["tokens_per_chunk"])

    chunks_all = []
    chunks_dict = {}
    web_page = []
    web_page_dict = {}

    i = 0
    counter = 0

    for _, row in df.iterrows():
        counter += 1
    
        if True:
        
            #tokens = tokenizer.encode(row['content_clean_translated'].strip(), add_special_tokens=False, return_tensors=None)
            #chunks = create_chunks(tokens, MAX_CHKS, tokenizer)
        
            chunks = text_splitter.create_documents([row['cleaned_content'].strip()])
            

            if i <= 10:
                print("\n---------------------------------\n")
                print("DOC", row['cleaned_content'])
                print()

            for chunk in chunks:
                chunks_all.append(chunk.page_content)
                chunks_dict[i] = chunk.page_content
                web_page.append(row['url'])
                web_page_dict[i] = row['url']

                if i <= 10:
                    print("CHUNK", i, chunk.page_content)
                    print()
                if i % 100 == 0:
                    print(i, "CHUNKS CREATED")
                i += 1

        
        # if i>100:
        #     break

    return chunks_all, chunks_dict, web_page_dict



