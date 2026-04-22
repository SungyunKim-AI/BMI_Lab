import pyautogui
import time
import numpy as np
from PIL import Image
from PIL import ImageChops


def wait_for_wizard_load(x, y, rgb, timeout=20):
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 마법사 창이 떴을 때만 나타나는 특정 위치의 픽셀 확인
        if pyautogui.pixelMatchesColor(x, y, rgb): 
            return True
        time.sleep(0.5) # 0.5초 간격으로 확인
    return False # 시간 초과 시 실패 리턴

def wait_for_save_confirm(timeout=20):
    start_time = time.time()
    while time.time() - start_time < timeout:
        # 마법사 창이 떴을 때만 나타나는 특정 위치의 픽셀 확인
        if pyautogui.pixelMatchesColor(798, 511, (252, 225, 0)): 
            return 'overload'
        elif pyautogui.pixelMatchesColor(756, 553, (240, 240, 240)) and pyautogui.pixelMatchesColor(1163, 594, (240, 240, 240)): 
            return 'save'  
        else:
            time.sleep(0.5) # 0.5초 간격으로 확인
    return 'Fail' # 시간 초과 시 실패 리턴

def extract(i):
    # 내보내기 결과 마법사 실행
    pyautogui.rightClick(630, 450 + 20 * i)
    time.sleep(0.3)
    pyautogui.press('down')
    time.sleep(0.2)
    pyautogui.press('enter')

    if wait_for_wizard_load(x=953, y=552, rgb=(0,0,0)):
        # 창이 확인되면 즉시 다음 단계 진행
        for j in range(5):
            pyautogui.click(955, 661)   # 확인 5번 클릭
            time.sleep(0.5)
        
        # 덮어쓰기 경고창
        wait_result = wait_for_save_confirm()
        if wait_result == 'overload':
            pyautogui.click(1030, 580)
            if wait_for_save_confirm() == 'save':
                pyautogui.click(1110, 575)
            return True
        elif wait_result == 'save':
            pyautogui.click(1110, 575)
            return True
        else:
            return False
    else:
        return False


if __name__ == '__main__':
    time.sleep(3)
    p = 1
    while True:
        # 환자 선택
        pyautogui.press('enter')

        for k in range(13): # 최대 방문건수 13회로 가정
            # 방문탭에 더 이상 방문이 없으면 종료
            time.sleep(1)
            pixel = pyautogui.screenshot().getpixel((490, 133 + 20 * k))
            if pixel == (255,255,255):
                break

            # 방문 선택
            pyautogui.click(490, 133 + 20 * k)
            time.sleep(2)
            pyautogui.click(1576, 216)
            time.sleep(1)

            # 테스트 내역 선택 및 추출
            cnt = 0
            last_idx = 25 # 목록의 가장 마지막 인덱스
            for i in range(26): 
                pixel = pyautogui.screenshot().getpixel((1192, 453 + 20 * i))
                if pixel != (255, 255, 255) and pixel != (240, 240, 240):
                    extract_result = extract(i)
                    time.sleep(0.5)
                    if extract_result:
                        print(f"#{p} patient #{k+1} visit #{cnt+1} data")
                    else:
                        print(f"Error: #{p} patient #{k+1} visit #{cnt+1} data")
                    cnt += 1
                        

            # 스크롤하며 추가 데이터 추출
            if cnt >= 24:                
                while True:
                    img_old = pyautogui.screenshot().crop((600, 500, 1000, 700))
                    pyautogui.press('down')
                    time.sleep(1)
                    img_new = pyautogui.screenshot().crop((600, 500, 1000, 700))
                    diff = ImageChops.difference(img_old, img_new)
                    
                    if not diff.getbbox():
                        break
                    
                    extract_result = extract(last_idx)
                    if extract_result:
                        print(f"#{p} patient #{k+1} visit #{cnt+1} data")
                    else:
                        print(f"Error: #{p} patient #{k+1} visit #{cnt+1} data")
                    cnt += 1
                    

        # 환자 목록으로 이동
        pyautogui.click(475, 1000)
        pyautogui.press('down')
        p += 1
