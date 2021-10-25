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
