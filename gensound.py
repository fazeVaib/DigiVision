import pyttsx3

def generate_sound(text):
    '''
    Converts text to audio and plays it.
    '''
    engine = pyttsx3.init() 
    engine.say(text)
    engine.runAndWait()