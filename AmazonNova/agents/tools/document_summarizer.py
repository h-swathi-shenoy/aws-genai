import boto3,json

bedrock_runtime_client = boto3.client("bedrock-runtime", region_name='us-east-1')
model_id_lite = "amazon.nova-lite-v1:0"


def document_summarizer_toolspec() -> json:
    
    '''Document Summarizer Tool Specificaitons'''
    
    return {"toolSpec": {
            "name": "Document_Summazier_Tool",
            "description": "Summary Generator tool for a given document in the given location",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Document path ",
                        },
                    },
                    "required": ["document_path"],
                }
            },
        }
    }



def document_summarizer(path:str) -> None:
    '''
    Tool to summarize a given document i.e. directly send the document to Bedrock without extracting the text in a document
    '''
    with open(path, "rb") as doc_file:
        doc_bytes = doc_file.read()
    
    doc_message = {
    "role": "user",
    "content": [
        {
            "document": {
                "name": "document",
                "format": "pdf",
                "source": {
                    "bytes": doc_bytes #no base64 encoding required!
                }
            }
        },
        { "text": "For a given document, give me the summary in the bulleted format. Summary generated should be formatted with markdown. Include images and tables as well in markdown" }
    ]
}
    response = bedrock_runtime_client.converse(
    modelId=model_id_lite,
    messages=[doc_message],
    inferenceConfig={
        "maxTokens": 2000,
        "temperature": 0
    },
)
    response_text = response['output']['message']['content'][0]['text']
    print(response_text)
    return