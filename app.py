import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
from util_funcs import get_length_param

def chat_function(message, length_of_the_answer, who_is_next, creativity):   # model, tokenizer
    
    input_user = message
    
    if length_of_the_answer == 'short':
        next_len = '1'
    elif length_of_the_answer == 'medium':
        next_len = '2'
    elif length_of_the_answer == 'long':
        next_len = '3'
    else:
        next_len = '-'
        
    print(who_is_next)
    if who_is_next == 'Kirill':
        next_who = 'G'
    elif who_is_next == 'Me':
        next_who = 'H'
        
        
    
    history = gr.get_state() or []
    chat_history_ids = torch.zeros((1, 0), dtype=torch.int) if history == [] else torch.tensor(history[-1][2], dtype=torch.long)

    # encode the new user input, add parameters and return a tensor in Pytorch
    if len(input_user) != 0:

        new_user_input_ids = tokenizer.encode(f"|0|{get_length_param(input_user, tokenizer)}|" \
                                              + input_user + tokenizer.eos_token, return_tensors="pt")
        # append the new user input tokens to the chat history
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        input_user = '-'
        
    if next_who == "G":

        # encode the new user input, add parameters and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(f"|1|{next_len}|", return_tensors="pt")
        # append the new user input tokens to the chat history
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

        print(tokenizer.decode(chat_history_ids[-1])) # uncomment to see full gpt input

        # save previous len
        input_len = chat_history_ids.shape[-1]
        # generated a response; PS you can read about the parameters at hf.co/blog/how-to-generate
        chat_history_ids = model.generate(
            chat_history_ids,
            num_return_sequences=1,                     # use for more variants, but have to print [i]
            max_length=512,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature = float(creativity),                          # 0 for greedy
            mask_token_id=tokenizer.mask_token_id,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            device='cpu'
        )

        response = tokenizer.decode(chat_history_ids[:, input_len:][0], skip_special_tokens=True)
    else:
        response = '-'
        
    history.append((input_user, response, chat_history_ids.tolist()))        
    gr.set_state(history)

    html = "<div class='chatbot'>"
    for user_msg, resp_msg, _ in history:
        if user_msg != '-':
            html += f"<div class='user_msg'>{user_msg}</div>"
        if resp_msg != '-':
            html += f"<div class='resp_msg'>{resp_msg}</div>"
    html += "</div>"
    return html





# Download checkpoint:
checkpoint = "Kirili4ik/ruDialoGpt3-medium-finetuned-telegram"
tokenizer =  AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
model = model.eval()

# Gradio
checkbox_group = gr.inputs.CheckboxGroup(['Kirill', 'Me'], default=['Kirill'], type="value", label=None)
title = "Chat with Kirill (in Russian)"
description = "Тут можно поболтать со мной. Но вместо меня бот. Оставь message пустым, чтобы Кирилл продолжил говорить. Подбробнее о технике по ссылке внизу."
article = "<p style='text-align: center'><a href='https://github.com/Kirili4ik/ruDialoGpt3-finetune-colab'>Github with fine-tuning GPT-2 on your chat</a></p>"
examples = [
            ["Привет, как дела?", 'medium', 'Kirill', 0.5],
            ["Сколько тебе лет?", 'medium', 'Kirill', 0.3],
]

iface = gr.Interface(chat_function,
                     [
                         "text",
                         gr.inputs.Radio(["short", "medium", "long"], default='medium'),
                         gr.inputs.Radio(["Kirill", "Me"], default='Kirill'),
                         gr.inputs.Slider(0, 1, default=0.5)
                     ],
                     "html",
                     title=title, description=description, article=article, examples=examples,
                     css= """
                            .chatbox {display:flex;flex-direction:column}
                            .user_msg, .resp_msg {padding:4px;margin-bottom:4px;border-radius:4px;width:80%}
                            .user_msg {background-color:cornflowerblue;color:white;align-self:start}
                            .resp_msg {background-color:lightgray;align-self:self-end}
                          """,
                     allow_screenshot=True,
                     allow_flagging=False
                    )

if __name__ == "__main__":
    iface.launch()
