# Image Search Engine
This repository contains the code to build an image search engine with [CLIP](https://openai.com/blog/clip/) and [FAISS](https://github.com/facebookresearch/faiss/wiki/)

If you want to know more about the project, please check out this blog

If you want a video walkthrough of the project, check out [this](https://www.youtube.com/watch?v=bCP-CHmn-DI) video

## Project Overview
* We will use CLIP to create an embedding for source images, the query image and also the query text
* We will add the embeddings to FAISS
* We will run a similarity search on embeddings with FAISS
* A live app will be deployed with Gradio

The final output would look like this
![app](https://github.com/Dhairya10/image-search-engine/blob/master/assets/app.png?raw=true)


## System Usage
I ran the code to sort through ~8k images on my local system. It took ~5 mins
The `htop` output is attached below for reference

![htop](https://github.com/Dhairya10/image-search-engine/blob/master/assets/htop.png?raw=true)