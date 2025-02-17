import boto3,json

bedrock_runtime_client = boto3.client("bedrock-runtime", region_name='us-east-1')
model_id_lite = "amazon.nova-lite-v1:0"


def image_summarizer_toolspec():
    
    '''Image Summarizer Tool Specificaitons'''
    
    return {"toolSpec": {
            "name": "Image_Summazier_Tool",
            "description": "Summary Generator tool for a given Image file present in the given location",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Image path ",
                        },
                    },
                    "required": ["image_path"],
                }
            },
        }
    }


def image_summarizer(path:str) -> None:
    '''
    Tool to summarize a given AWS Architecture i.e. directly send the image path to Bedrock without extracting the opening the image
    '''
    with open(path, "rb") as image_file:
        image_bytes = image_file.read()
    
    image_message = {
    "role": "user",
    "content": [
        { "text": "Image 1:" },
        {
            "image": {
                "format": "png",
                "source": {
                    "bytes": image_bytes #no base64 encoding required!
                }
            }
        },
        { "text": "You are an AWS Cloud Architect. When the user provides you with an Architecture Diagram, describe the workflow and also suggest how it can be helpful" }
    ],
}

    response = bedrock_runtime_client.converse(
    modelId=model_id_lite,
    messages=[image_message],
    inferenceConfig={
        "maxTokens": 2000,
        "temperature": 0
    },
)
    response_text = response['output']['message']['content'][0]['text']
    print(response_text)
    return
    
    