#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pygame #파이 게임 모듈 임포트
import random
import time


SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1000

pygame.init() #파이 게임 초기화
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT)) #화면 크기 설정
pygame.display.set_caption("출석체크")
clock = pygame.time.Clock() 

#색깔 변수

BLACK = (0, 0, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)
WHITE = (255, 255,255)

#폰트
large_font = pygame.font.SysFont('malgungothic', 54)
small_font = pygame.font.SysFont('malgungothic', 36)

start_time = int(time.time()) #1970년 1월 1일 0시 0분 0초 부터 현재까지 초
remain_second = 30
game_over = False

phonix_image = pygame.image.load('phonix_tracking.jpg')
phonix_image = pygame.transform.scale(phonix_image, (100, 102))

phonixs=[]
li = [1,2]
choiceList = random.choice(li)
#     phonix = phonix_image.get_rect(left=random.randint(0, SCREEN_WIDTH) - phonix_image.get_width(), 
#                                    top=random.randint(0, SCREEN_HEIGHT) - phonix_image.get_height())
if choiceList == 1:
    phonix = phonix_image.get_rect(left=100 - phonix_image.get_width(), top=500 - phonix_image.get_height())
    phonixs.append(phonix)
else: 
    phonix = phonix_image.get_rect(right=1600 - phonix_image.get_width(), top=500 - phonix_image.get_height())
    phonixs.append(phonix)

while True: #게임 루프
    screen.fill(WHITE) #단색으로 채워 화면 지우기
    
    text_Title= small_font.render("포닉스를 눈으로 따라가세요!", True, BLACK)
    # Rect 생성
    text_Rect = text_Title.get_rect()

    # 가로 가운데, 세로 50 위치
    text_Rect.centerx = round(SCREEN_WIDTH / 2)
    text_Rect.y = 20

    # Text Surface SCREEN에 복사하기, Rect 사용
    screen.blit(text_Title, text_Rect)

    #변수 업데이트
    event = pygame.event.poll() #이벤트 처리
    
    
    if event.type == pygame.QUIT:
        break
    elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
        print(event.pos[0], event.pos[1])  #포닉스의 이동좌표

        
    
    if choiceList == 1:
        phonix.right += 5  #오른쪽으로 5씩 이동
    else:
        phonix.left -= 5  # 왼쪽으로 5씩 이동
        
    for phonix in phonixs:
        if not phonix.colliderect(screen.get_rect()):
            phonixs.remove(phonix)
            if choiceList == 1:
                phonix1 = phonix_image.get_rect(left=100 - phonix_image.get_width(), top=500 - phonix_image.get_height())
#                 phonixs.append(phonix1)
            else: 
                phonix2 = phonix_image.get_rect(right=1600 - phonix_image.get_width(), top=500 - phonix_image.get_height())
#                 phonixs.append(phonix2)
        
    if not game_over:
        current_time = int(time.time())
        remain_second = 11 - (current_time - start_time)

        if remain_second <= 0:
            game_over = True
            
            

    #화면 그리기

    for phonix in phonixs: 
        screen.blit(phonix_image, phonix) 

    remain_second_image = small_font.render('남은 시간 {}'.format(remain_second), True, YELLOW)
    screen.blit(remain_second_image, remain_second_image.get_rect(right=SCREEN_WIDTH - 10, top=10))

    if game_over:
        game_over_image = large_font.render('출석확인', True, RED)
        screen.blit(game_over_image, game_over_image.get_rect(centerx=SCREEN_WIDTH // 2, centery=SCREEN_HEIGHT // 2))
    

    pygame.display.update() #모든 화면 그리기 업데이트
    clock.tick(30) #30 FPS (초당 프레임 수) 를 위한 딜레이 추가, 딜레이 시간이 아닌 목표로 하는 FPS 값

pygame.quit() 


# In[ ]:




