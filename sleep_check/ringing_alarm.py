
# coding: utf-8

# In[ ]:


import pygame

def select_alarm(result) :
    if result == 0:
        sound_alarm("wake_up_sound.wav")

def sound_alarm(path) :
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()
    

