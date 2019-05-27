from pygame import mixer
from tempfile import TemporaryFile
from gtts import gTTS

def generate_sound(text):
    '''
    Converts text to audio and plays it.
    '''
    language = 'en'
    myobj = gTTS(text=text, lang=language, slow=False)
    # slow = False for high speed
    sf = TemporaryFile()
    myobj.write_to_fp(sf)
    sf.seek(0)
    mixer.init()
    mixer.music.load(sf)
    mixer.music.play()
