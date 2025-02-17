import boto3, json
import base64
import io
import sys
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

bedrock_runtime_client = boto3.client("bedrock-runtime", region_name='us-east-1')
model_id_canvas = "amazon.nova-canvas-v1:0"

def country_canvas_toolspec() -> json:
    
    '''Canvas Tool Specificaitons'''
    
    return { "toolSpec": {
            "name": "Country_Canvas_cTool",
            "description": "Generate the image of the given country,representing the most typical country's characteristics,incorporating its flag.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description":"Country for the image generator tool ",
                        },
                    },
                    "required": ["query"],
                }
            },
        }
    }


def plot_images(base_images, prompt=None, seed=None, ref_image_path=None, color_codes=None, original_title=None, processed_title=None):
    if ref_image_path and color_codes:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        num_subplots = 3
    elif ref_image_path or color_codes:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        num_subplots = 2
    else:
        fig, axes = plt.subplots(1, 1, figsize=(6, 5))
        num_subplots = 1
    
    axes = np.array(axes).ravel() 
    
    current_subplot = 0
    
    if color_codes:
        num_colors = len(color_codes)
        color_width = 0.8 / num_colors
        for i, color_code in enumerate(color_codes):
            x = i * color_width
            rect = plt.Rectangle((x, 0), color_width, 1, facecolor=f'{color_code}', edgecolor='white')
            axes[current_subplot].add_patch(rect)
        axes[current_subplot].set_xlim(0, 0.8)
        axes[current_subplot].set_ylim(0, 1)
        axes[current_subplot].set_title('Color Codes')
        axes[current_subplot].axis('off')
        current_subplot += 1
    
    if ref_image_path:
        reference_image = Image.open(ref_image_path)
        max_size = (512, 512)
        reference_image.thumbnail(max_size)
        axes[current_subplot].imshow(np.array(reference_image))
        axes[current_subplot].set_title(original_title or 'Reference Image')
        axes[current_subplot].axis('off')
        current_subplot += 1
    
    axes[current_subplot].imshow(np.array(base_images[0]))
    if processed_title:
        axes[current_subplot].set_title(processed_title)
    elif ref_image_path and seed is not None:
        axes[current_subplot].set_title(f'Image Generated Based on Reference\nSeed: {seed}')
    elif seed is not None:
        axes[current_subplot].set_title(f'Image Generated\nSeed: {seed}')
    else:
        axes[current_subplot].set_title('Processed Image')
    axes[current_subplot].axis('off')
    
    if prompt:
        print(f"Prompt: {prompt}\n")
    
    plt.tight_layout()
    plt.show()

    
def country_canvas(country:str):
    
    ''' Generate the Creative Image for a given Country'''
    
    prompt= f"You generate image of a country representing the most typical country's characteristics,\
        incorporating its flag. the country is {country}" 

    body = json.dumps(
        {
            "taskType": "TEXT_IMAGE",
            "textToImageParams": {
                "text": prompt,                    # Required
                #"negativeText": negative_prompts   # Optional
            },
            "imageGenerationConfig": {
                "numberOfImages": 1,   # Range: 1 to 5 
                "quality": "standard",  # Options: standard or premium
                "height": 1024,        
                "width": 1024,         
                "cfgScale": 7.5,       # Range: 1.0 (exclusive) to 10.0
                "seed": 250 #100            # Range: 0 to 214783647
            }
        }
    )

    response = bedrock_runtime_client.invoke_model(
        body=body, 
        modelId=model_id_canvas,
        accept="application/json", 
        contentType="application/json"
    )

    response_body = json.loads(response.get("body").read())
    response_images = [
        Image.open(io.BytesIO(base64.b64decode(base64_image)))
        for base64_image in response_body.get("images")
    ]
    return plot_images(response_images, processed_title=f"AI Image for country{country} ") 