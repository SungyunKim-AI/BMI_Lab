import pyautogui
import time
import numpy as np
from PIL import Image
from PIL import ImageChops


def extract(i):
    # 내보내기 결과 마법사 실행
    pyautogui.rightClick(630, 450 + 20 * i)
    pyautogui.press('down')
    pyautogui.press('enter')

    # 확인 5번 클릭
    for j in range(5):
        pyautogui.click(955, 661)
        time.sleep(0.5)
    time.sleep(1)
    
    # 최대 20초 동안 확인 버튼 뜨는지 확인
    wait = 0
    check_x_coords = [790, 792, 794, 796, 798, 800, 802, 804, 806, 808, 810]
    while (wait < 20) and (pyautogui.screenshot().getpixel((964, 555)) != (240,240,240)):
        screenshot = pyautogui.screenshot()
        if any(180 < screenshot.getpixel((x, 507))[1] < 220 for x in check_x_coords):
            pyautogui.click(1028, 566)
        time.sleep(1)
        wait += 1

    # 확인 버튼 또는 덮어쓰기 버튼 한번 더 클릭
    screenshot = pyautogui.screenshot()
    if any(180 < screenshot.getpixel((x, 507))[1] < 220 for x in check_x_coords):
        pyautogui.click(1028, 566)
        pyautogui.click(1030, 580)
    time.sleep(3)
    pyautogui.click(1100, 563)
    time.sleep(1)


if __name__ == '__main__':
    time.sleep(3)
    p = 1
    while True:
        # 환자 선택
        pyautogui.press('enter')

        for k in range(13): # 최대 방문건수 13회로 가정
            # 방문탭에 더 이상 방문이 없으면 종료
            pixel = pyautogui.screenshot().getpixel((490, 133 + 20 * k))
            if pixel == (255,255,255):
                break

            # 방문 선택
            pyautogui.click(490, 133 + 20 * k)
            time.sleep(2)
            pyautogui.click(1576, 216)
            time.sleep(0.5)

            # 테스트 내역 선택 및 추출
            cnt = 1
            for i in range(25): # 최대 테스트 내역 25건으로 가정
                pixel = pyautogui.screenshot().getpixel((1192, 453 + 20 * i))
                if pixel != (255,255,255) and pixel != (240,240,240):
                    extract(i)
                    print(f"#{p} patient #{k+1} visit #{cnt} data")
                    cnt += 1

            # 스크롤 전/후 비교해서 남은 데이터가 있는지 확인
            if cnt >= 23:
                img_old = pyautogui.screenshot().crop((600, 500, 1000, 700))
                pyautogui.press('down')
                img_new = pyautogui.screenshot().crop((600, 500, 1000, 700))
                diff = ImageChops.difference(img_old, img_new)
                while diff.getbbox():
                    img_old = img_new
                    extract(24)
                    cnt += 1
                    print(f"#{p} patient #{k+1} visit #{cnt} data")
                    pyautogui.press('down')
                    img_new = pyautogui.screenshot().crop((600, 500, 1000, 700))
                    diff = ImageChops.difference(img_old, img_new)

        # 환자 목록으로 이동
        pyautogui.click(475, 1000)
        pyautogui.press('down')
        p += 1

