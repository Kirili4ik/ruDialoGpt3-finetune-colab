import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_length_param(text: str, tokenizer) -> str:
    """Maps text to 1 of 4 buckets based on length after encoding.

    Parameters
    ----------
    text: str
        The text to be given 1 of 4 length parameters.

    tokenizer: HuggingFace tokenizer 
        Tokenizer that used to compute the length of the text after encoding.
        For more info ee https://huggingface.co/transformers/main_classes/tokenizer.html

    Returns
    -------
    len_param: str
        One of four buckets: 
        '1' for short, '2' for medium, '3' for long texts and '-' for all others. 
    """
    tokens_count = len(tokenizer.encode(text))
    if tokens_count <= 15:
        len_param = '1'
    elif tokens_count <= 50:
        len_param = '2'
    elif tokens_count <= 256:
        len_param = '3'
    else:
        len_param = '-'
    return len_param


def get_user_param(text: dict, machine_name_in_chat: str) -> str:
    """Maps text by 1/0 for it to be the person or the machine in the dialogue

    Parameters
    ----------
    text: Dict[..., 'from', ...]
        Dict containing field 'from' with the name of the user who sent the message

    machine_name_in_chat: str
        Str with the name of the machine - it will be predicted
    """
    if text['from'] == machine_name_in_chat:
        return '1'  # machine
    else:
        return '0'  # human


def build_text_file(data_json: dict, dest_path: str, 
                    tokenizer, machine_name_in_chat='Кирилл Гельван'):
    """Create a text file for training in special format for ruDialoGPT-3.

    Parameters
    ----------
    data_json: dict
        Dict containing 'text' (message) and 'from' (user who sent the message)
        
    dest_path: str
        String containing path to write data there

    tokenizer: HuggingFace tokenizer 
        Tokenizer that used to compute the length of the text after encoding.
        For more info ee https://huggingface.co/transformers/main_classes/tokenizer.html
    """
    f = open(dest_path, 'w')
    new_data = ''
    for i in range(len(data_json) - 1):
        message, next_message = data_json[i], data_json[i+1]
        if message['text'] == '' or type(message['text']) != str:
            continue
        if next_message['text'] == '' or type(next_message['text']) != str:
            continue

        user   = get_user_param(message, machine_name_in_chat=machine_name_in_chat)
        length = get_length_param(data_json[i+1]['text'], tokenizer)
        message_text = re.sub(r"\n", ". ", message['text'])
        new_data += f"|{user}|{length}|{message_text}{tokenizer.eos_token}" + "\n"

    f.write(new_data)


def load_dataset(train_path, test_path, tokenizer):
    """Creates train and test PyTorch datasets and collate_fn using HuggingFace.

    Parameters
    ----------
    train_path: str
        String containing path to train data
        
    test_path: str
        String containing path to test data

    tokenizer: HuggingFace tokenizer 
        Tokenizer that used to compute the length of the text after encoding.
        For more info ee https://huggingface.co/transformers/main_classes/tokenizer.html
    """
    train_dataset = TextDataset(
          tokenizer  = tokenizer,
          file_path  = train_path,
          block_size = 256)
     
    test_dataset = TextDataset(
          tokenizer  = tokenizer,
          file_path  = test_path,
          block_size = 256)   
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    return train_dataset, test_dataset, data_collator


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

    #########     next_who = input("Who's phrase?\t")  #input("H / G?")     # Human or GPT

    # In case Human
    ##### if next_who == "H":

    ########    input_user = input("===> Human: ")
    # encode the new user input, add parameters and return a tensor in Pytorch
    if len(input_user) != 0:

        new_user_input_ids = tokenizer.encode(f"|0|{get_length_param(input_user, tokenizer)}|" \
                                              + input_user + tokenizer.eos_token, return_tensors="pt")
        # append the new user input tokens to the chat history
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
    else:
        input_user = '-'
        
    if next_who == "G":

        ######## next_len = input("Phrase len? 1/2/3/-\t")  #input("Exp. len?(-/1/2/3): ")
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


checkbox_group = gr.inputs.CheckboxGroup(['Kirill', 'Me'], default=['Kirill'], type="value", label=None)

inputs = gr.inputs.Textbox(lines=1, label="???")
outputs =  gr.outputs.Textbox(label="Kirill (GPT-2):")
title = "Chat with Kirill (in Russian)"
description = "Тут можно поболтать со мной. Но вместо меня бот. Оставь message пустым, чтобы Кирилл продолжил говорить. Подбробнее о технике по ссылке внизу."
article = "<p style='text-align: center'><a href='https://github.com/Kirili4ik/ruDialoGpt3-finetune-colab'>Github with fine-tuning GPT-2 on your chat</a></p>"
examples = [
            ["Привет, как дела?", 'medium', 'Kirill', 0.6],
            ["Сколько тебе лет?", 'medium', 'Kirill', 0.3],
]

iface = gr.Interface(chat_function, 
                     [    
                         "text", 
                         gr.inputs.Radio(["short", "medium", "long"], default='medium'), 
                         gr.inputs.Radio(["Kirill", "Me"], default='Kirill'),
                         gr.inputs.Slider(0, 1, default=0.6)
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

iface.launch()
