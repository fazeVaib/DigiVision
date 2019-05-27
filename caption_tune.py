import random

def face_found_cap(text):
    cap_list = [
        'Hey, it is ' + text + '. Say Hello!',
        'I see ' + text + ' right there in front of you',
        'I know who the person is. It is ' + text
    ]

    return random.choice(cap_list)

def face_not_found_cap():
    cap_list = [
        'I bet I have never seen this person in my life before',
        'No, I do not know who this person is.',
        'Unknown person alert.'
    ]

    return random.choice(cap_list)

def modcap(text):
    text = text[:-4]
    text = text.strip()
    if text == "a man in a suit and tie holding a glass of wine":
        return "a person is there in front of you"
    else:
        return text
