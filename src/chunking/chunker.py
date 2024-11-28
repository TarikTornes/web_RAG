from text_splitter import chunker

def chunk_data(df):
    """This function is supposed to chunk data into smaller text blocks

    @param: df represents the dataframe that contains the data/documents to be chunked
            in the column "cleaned_content"
    @return: list of all chunks created in order
    @return: dict where the key is the index of the chunk in the list, and the value is the chunk itself
    """

    chunks_all = []
    chunks_dict = {}

    i = 0
    counter = 0

    for idx, row in df.iterrows():
        print(counter)
        counter += 1
    
        if True:
        
            #tokens = tokenizer.encode(row['content_clean_translated'].strip(), add_special_tokens=False, return_tensors=None)
            #chunks = create_chunks(tokens, MAX_CHKS, tokenizer)
        
            chunks = chunker(row['cleaned_content'])

            if i <= 10:
                print("\n---------------------------------\n")
                print("DOC", row['cleaned_content'])
                print()

            for chunk in chunks:
                chunks_all.append(chunk.page_content)
                chunks_dict[i] = chunk.page_content
                if i <= 10:
                    print("CHUNK", i, chunk.page_content)
                    print()
                if i % 100 == 0:
                    print(i, "CHUNKS CREATED")
                i += 1

    return chunks_all, chunks_dict
        #if i>200:
        #    break



