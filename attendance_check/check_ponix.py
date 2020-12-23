import pygame
import random
import time
import sys
import numpy as np

class moving_ponix(pygame.sprite.Sprite):

    SCREEN_WIDTH = 1000
    SCREEN_HEIGHT = 400
    FPS = 20
    
    # pygame 초기화
    pygame.init()
    
    #폰트
    large_font = pygame.font.SysFont('Applegothic', 36)
    small_font = pygame.font.SysFont('Applegothic', 24)
    remain_second = 30
    
    # 색상
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 255, 0)
    WHITE = (255, 255,255)
    navy_blue = ((0,0,100))

    
    def __init__(self):
        self.ponix_pos = None   # 포닉스의 위치 정보
        self.image1 = pygame.transform.scale(pygame.image.load('ponix1.jpg'),
                                             (50, 51))
        self.image2 = pygame.transform.scale(pygame.image.load('ponix2.jpg'),
                                             (50, 51))
        self.count = 0      # 몇번이나 맞췄는지 count
        self.num = 0
        
        self.game_over = False
        
        self.screen = None
        self.clock = None
        
        self.start_time = int(time.time()) #1970년 1월 1일 0시 0분 0초 부터 현재까지 초

    
    def set_ponix(self):
        self.screen = pygame.display.set_mode((self.SCREEN_WIDTH, self.SCREEN_HEIGHT))
        self.clock = pygame.time.Clock()
        pygame.display.set_caption('출석체크')
            
        li = [0,1]
        choice = random.choice(li)
        
        # 왼쪽 출발
        if not choice:
            center = -100
            top = 200
            speed = 1
            self.ponix_pos = [center, top, speed]
        # 오른쪽 출발
        else:
            center = 1100
            top = 200
            speed = -1
            self.ponix_pos = [center, top, speed]         
    
    
    # 포닉스의 위치와 눈의 위치가 같은지 확인
    def check_eye(self, flag):
        
        # 출석체크 확인
        if flag:
            game_over_image = self.large_font.render('출석확인', True, self.RED)
            self.screen.blit(game_over_image,
                             game_over_image.get_rect(centerx=self.SCREEN_WIDTH // 2,
                                                      centery=self.SCREEN_HEIGHT // 2))
            
        else:
            game_over_image = self.large_font.render('출석실패', True, self.RED)
            self.screen.blit(game_over_image,
                             game_over_image.get_rect(centerx=self.SCREEN_WIDTH // 2,
                                                      centery=self.SCREEN_HEIGHT // 2))
            
        pygame.display.update()
    
    def sign(self, direct, flag):
        # True 인 경우
        if flag == True:
            if direct == 'Right':
                text_OK = self.small_font.render("OK",True, self.RED)
                text_Rect = text_OK.get_rect()
                text_Rect.centerx = round(self.SCREEN_WIDTH * 3/ 4)
                text_Rect.y = 50
                self.screen.blit(text_OK, text_Rect)

                self.count += 1
            else:
                text_Look = self.small_font.render("Look at the Ponix",True, self.RED)
                text_Rect = text_Look.get_rect()
                text_Rect.centerx = round(self.SCREEN_WIDTH * 1/ 4)
                text_Rect.y = 50
                self.screen.blit(text_Look, text_Rect)
        
        else:
            if direct == 'Left':
                text_OK = self.small_font.render("OK",True, self.RED)
                text_Rect = text_OK.get_rect()
                text_Rect.centerx = round(self.SCREEN_WIDTH * 1/ 4)
                text_Rect.y = 50
                self.screen.blit(text_OK, text_Rect)  
                
                self.count += 1
            else:
                text_Look = self.small_font.render("Look at the Ponix",True, self.RED)
                text_Rect = text_Look.get_rect()
                text_Rect.centerx = round(self.SCREEN_WIDTH * 3/ 4)
                text_Rect.y = 50
                self.screen.blit(text_Look, text_Rect)

        pygame.display.update()  # 모든 화면 그리기 업데이트

    # 포닉스 이동
    def update(self):
        pass
        

    def end_game(self):         
        pygame.quit()



    # 게임 실행
    def start_game(self):     
        
        remain_second = 0
        
        self.screen.fill(self.WHITE)
        text_Title = self.small_font.render("포닉스를 눈으로 따라가세요!",
                                            True, self.BLACK)
        
        # 포닉스 생성
        if self.num:
            ponix = self.image1.get_rect(centerx = self.ponix_pos[0],
                                     centery = self.ponix_pos[1])
        else:
            ponix = self.image2.get_rect(centerx = self.ponix_pos[0],
                                    centery = self.ponix_pos[1])
        
        # Rect 생성
        text_Rect = text_Title.get_rect()
        text_Rect.centerx = round(self.SCREEN_WIDTH / 2)
        text_Rect.y = 20
        
        self.screen.blit(text_Title, text_Rect)
        
        ponix.centerx += self.ponix_pos[2]*np.random.randint(2,10)
        self.ponix_pos[0] = ponix.centerx
        
        for event in pygame.event.get():
            if event.type==pygame.QUIT:
                pygame.quit()
                sys.exit()
        
             
        if not self.game_over:
            current_time = int(time.time())
            remain_second = 9 - (current_time - self.start_time)
            
            if remain_second <= 0:
                self.game_over = True
        
        
        if self.num:
            self.screen.blit(self.image1, ponix)
            self.num = 0
        else:
            self.screen.blit(self.image2, ponix)
            self.num = 1
            
        remain_second_image = self.small_font.render("출석 확인 중 ... ",
                                                True,
                                                self.navy_blue)
        
        self.screen.blit(remain_second_image,
                         remain_second_image.get_rect(right=self.SCREEN_WIDTH - 10,
                                                      top=10))
        
        
        pygame.display.update() #모든 화면 그리기 업데이트
        self.clock.tick(30) #30 FPS (초당 프레임 수) 를 위한 딜레이 추가, 딜레이 시간이 아닌 목표로 하는 FPS 값
        
        return ponix.centerx 