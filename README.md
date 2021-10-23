# ruDialoGpt3 colab for finetuning on telegram chat
This is a colab-ready-for-use tutorial for finetuning ruDialoGpt3 model on your telegram chat using HuggingFace and PyTorch.

- 🤗 [Model page](https://huggingface.co/Kirili4ik/ruDialoGpt3-medium-finetuned-telegram) 

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fnAVURjyZRK9VQg1Co_-SKUQnRES8l9R?usp=sharing) Colab tutorial 


I used RuDialoGPT-3 trained on forums to fine tune. It was trained by [@Grossmend](https://github.com/Grossmend) on Russian forums. The training procedure of the model for dialogue is described in Grossmend's [blogpost](https://habr.com/ru/company/icl_services/blog/548244/) (in Russian). **I have created a simple pipeline and fine tuned that model on my own exported telegram chat (~30mb json, 3 hours of fine tuning**). It is in fact very easy to get the data from telegram and fine tune a model:

1) Export your telegram chat

![](https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/how-to-export-chat.jpg)

2) Upload it to colab

![](https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/how-to-upload-json.jpg)

3) The code will create a dataset for you

4) Wait a bit! 
 
5) :tada: (Inference and smile)

Or you can just go to google colab and play with my finetuned model!:

<details>
  <summary><b>A couple of dialogue samples:</b>
  </summary>
  <img src="https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/sample1.jpg">
  <img src="https://raw.githubusercontent.com/Kirili4ik/ruDialoGpt3-finetune-colab/main/sample2.jpg">
</details>


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fnAVURjyZRK9VQg1Co_-SKUQnRES8l9R?usp=sharing#scrollTo=psXZnJk0Eo3J) Inference part
