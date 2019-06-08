import random

def face_found_cap(text):
    cap_list = [
        'Hey, it is ' + text + '. Say Hello!',
        'I see ' + text + '. Go and say Hello!',
        'I see a familiar face. It seems to be ' + text
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
        return "a person standing just in front."
    else:
        return text
