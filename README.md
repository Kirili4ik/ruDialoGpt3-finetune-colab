# ruDialoGpt3 colab for finetuning on telegram chat
This is a ready-for-use-colab tutorial for finetuning ruDialoGpt3 model on your telegram chat using HuggingFace and PyTorch

Here is a picture of a demo you can also setup with Gradio and Huggingface Spaces:

![t](https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/samples/sample6.png)

- 🤗 [Spaces interactable demo](https://huggingface.co/spaces/Kirili4ik/chat-with-Kirill) of my model using [Gradio](https://github.com/gradio-app/gradio)

- 🤗 [Model page](https://huggingface.co/Kirili4ik/ruDialoGpt3-medium-finetuned-telegram) 

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fnAVURjyZRK9VQg1Co_-SKUQnRES8l9R?usp=sharing) Colab tutorial 


I used RuDialoGPT-3 trained on forums to fine tune. It was trained by [@Grossmend](https://github.com/Grossmend) on Russian forums. The training procedure of the model for dialogue is described in Grossmend's [blogpost](https://habr.com/ru/company/icl_services/blog/548244/) (in Russian). **I have created a simple pipeline and fine tuned that model on my own exported telegram chat (~30mb json, 3 hours of fine tuning**). 

It is in fact very easy to get the data from telegram and fine tune a model:
<details>
 <summary><b>Using telegram chat for fine-tuning</b>
 </summary>
1) Export your telegram chat as JSON

![](https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/samples/how-to-export-chat.jpg)

2) Upload it to colab

![](https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/samples/how-to-upload-json.jpg)

3) The code will create a dataset for you

4) Wait a bit! 
 
5) :tada: (Inference and smile)
</details>
 
> To run using 🤗Spaces or [Gradio](https://github.com/gradio-app/gradio) check [app.py](app.py)  
 

<details>
  <summary><b>A couple of dialogue samples</b>
  </summary> 
  <img src="https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/samples/sample3.jpg">
  <img src="https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/samples/sample4.jpg">
  <img src="https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/samples/sample5.jpg">
  <img src="https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/samples/sample1.jpg">
  <img src="https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/samples/sample2.jpg">
</details>
