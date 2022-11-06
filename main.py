import torch
import clip
from PIL import Image
import faiss 
import os
import numpy as np
import gradio as gr
import shutil


# Create output directory, if it does not exist
output_dir = os.path.join(os.getcwd(), "output")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialising CLIP
device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


def add_embed_to_faiss(image_embed):
    # FAISS Parameters
    d = 512
    nlist = 2
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)

    assert not index.is_trained
    index.train(image_embed)
    assert index.is_trained
    index.add(image_embed)
    
    return index


def get_image_embed(image):
    image = preprocess(Image.open(image)).unsqueeze(0).to(device)

    with torch.no_grad():
        image_embed = model.encode_image(image)
        image_embed = image_embed / image_embed.norm(dim=1, keepdim=True)

    # Output is converted to numpy for feeding into FAISS
    return image_embed.numpy()


def get_text_embed(text):
    text = clip.tokenize(text).to(device)
    with torch.no_grad():
        text_embed = model.encode_text(text)
        text_embed = text_embed / text_embed.norm(dim=1, keepdim=True)

    # Output is converted to numpy for feeding into FAISS
    return text_embed.numpy()


def query_faiss(query_embed, num_images, index):
    index.nprobe = 1
    distance_list, index_list = index.search(query_embed, num_images)

    return distance_list, index_list


def process_faiss_output(index_list, file_obj):
    for idx in index_list[0]:
        file_name = file_obj[idx].name
        shutil.move(file_name, f'{output_dir}/{file_name.split("/")[-1]}.jpg')


def process_images(image_list):
    embed_list = []
    for image_file in image_list:
        try:
            image_embed = get_image_embed(image_file.name)
            embed_list.append(image_embed)
        except Exception as e:
            print(f"Exception : {e}")

    return embed_list


def process_gradio_text(file_obj, query_text, num_images):

    if num_images > len(file_obj):
        return f"[ERROR] Num of query images is greater than total images"

    embed_list = process_images(file_obj)
    embed_array = np.vstack(embed_list)
    index = add_embed_to_faiss(embed_array)
    text_embed = get_text_embed(query_text)
    _, index_list = query_faiss(text_embed, int(num_images), index)
    process_faiss_output(index_list, file_obj)     

    return output_dir


def process_gradio_image(file_obj, query_image, num_images):

    if num_images > len(file_obj):
        return f"[ERROR] Num of query images is greater than total images"

    print(type(query_image))
    embed_list = process_images(file_obj)
    embed_array = np.vstack(embed_list)
    index = add_embed_to_faiss(embed_array)
    image_embed = get_image_embed(query_image)
    _, index_list = query_faiss(image_embed, int(num_images), index)
    process_faiss_output(index_list, file_obj)     

    return output_dir


if __name__=="__main__":

    with gr.Blocks() as demo:

        with gr.Tab("Text Search"):
            text_input_file = gr.File(file_count='directory', label="File Picker")
            text_query = gr.Textbox(label="Query Text")
            num_images_tq = gr.Slider(minimum=1, maximum=10000, step=10.0, label="Num Images")
            text_output = gr.Textbox(label="Output Directory")
            text_button = gr.Button("Find")

        with gr.Tab("Image Search"):
            image_input_file = gr.File(file_count='directory', label="File Picker")
            image_input = gr.File(label="Input Image")
            num_images_iq = gr.Slider(minimum=1, maximum=10000, step=10.0, label="Num Images")
            image_output = gr.Textbox(label="Output Directory")
            image_button = gr.Button("Find")

        text_button.click(process_gradio_text, inputs=[text_input_file, text_query, num_images_tq], outputs=text_output)
        image_button.click(process_gradio_image, inputs=[image_input_file, image_input, num_images_iq], outputs=image_output)

    demo.launch()