import pyautogui
import time
import numpy as np
from PIL import Image
from PIL import ImageChops


def extract(i):
    pyautogui.click(630, 450 + 20 * i, button='right')
    pyautogui.press('down')
    pyautogui.press('enter')
    time.sleep(1)
    for j in range(5):
        pyautogui.click(955, 661)
        time.sleep(0.5)
    time.sleep(1)
    wait = 1
    while not np.array_equiv(pyautogui.screenshot().getpixel((964, 555)), [240, 240, 240]):
        if (180 < pyautogui.screenshot().getpixel((790, 507))[1] and pyautogui.screenshot().getpixel((790, 507))[1] < 220) \
            or ((180 < pyautogui.screenshot().getpixel((792, 507))[1] and pyautogui.screenshot().getpixel((792, 507))[1] < 220)) \
            or (180 < pyautogui.screenshot().getpixel((794, 507))[1] and pyautogui.screenshot().getpixel((794, 507))[1] < 220) \
            or (180 < pyautogui.screenshot().getpixel((796, 507))[1] and pyautogui.screenshot().getpixel((796, 507))[1] < 220) \
            or (180 < pyautogui.screenshot().getpixel((798, 507))[1] and pyautogui.screenshot().getpixel((798, 507))[1] < 220) \
            or (180 < pyautogui.screenshot().getpixel((800, 507))[1] and pyautogui.screenshot().getpixel((800, 507))[1] < 220) \
            or (180 < pyautogui.screenshot().getpixel((802, 507))[1] and pyautogui.screenshot().getpixel((802, 507))[1] < 220) \
            or (180 < pyautogui.screenshot().getpixel((804, 507))[1] and pyautogui.screenshot().getpixel((804, 507))[1] < 220) \
            or (180 < pyautogui.screenshot().getpixel((806, 507))[1] and pyautogui.screenshot().getpixel((806, 507))[1] < 220) \
            or (180 < pyautogui.screenshot().getpixel((808, 507))[1] and pyautogui.screenshot().getpixel((808, 507))[1] < 220) \
            or (180 < pyautogui.screenshot().getpixel((810, 507))[1] and pyautogui.screenshot().getpixel((810, 507))[1] < 220):
            pyautogui.click(1028, 566)
        time.sleep(1)
        wait += 1
        if wait >= 20:
            break
    if (180 < pyautogui.screenshot().getpixel((790, 507))[1] and pyautogui.screenshot().getpixel((790, 507))[1] < 220) \
        or ((180 < pyautogui.screenshot().getpixel((792, 507))[1] and pyautogui.screenshot().getpixel((792, 507))[1] < 220)) \
        or (180 < pyautogui.screenshot().getpixel((794, 507))[1] and pyautogui.screenshot().getpixel((794, 507))[1] < 220) \
        or (180 < pyautogui.screenshot().getpixel((796, 507))[1] and pyautogui.screenshot().getpixel((796, 507))[1] < 220) \
        or (180 < pyautogui.screenshot().getpixel((798, 507))[1] and pyautogui.screenshot().getpixel((798, 507))[1] < 220) \
        or (180 < pyautogui.screenshot().getpixel((800, 507))[1] and pyautogui.screenshot().getpixel((800, 507))[1] < 220) \
        or (180 < pyautogui.screenshot().getpixel((802, 507))[1] and pyautogui.screenshot().getpixel((802, 507))[1] < 220) \
        or (180 < pyautogui.screenshot().getpixel((804, 507))[1] and pyautogui.screenshot().getpixel((804, 507))[1] < 220) \
        or (180 < pyautogui.screenshot().getpixel((806, 507))[1] and pyautogui.screenshot().getpixel((806, 507))[1] < 220) \
        or (180 < pyautogui.screenshot().getpixel((808, 507))[1] and pyautogui.screenshot().getpixel((808, 507))[1] < 220) \
        or (180 < pyautogui.screenshot().getpixel((810, 507))[1] and pyautogui.screenshot().getpixel((810, 507))[1] < 220):
        pyautogui.click(1028, 566)
    time.sleep(2)
    wait += 2
    pyautogui.click(1100, 563)
    time.sleep(1)


if __name__ == '__main__':
    time.sleep(3)
    p = 0
    while True:
        pyautogui.press('enter')
        p += 1
        k = 0
        while k <= 12:
            if (pyautogui.screenshot().getpixel((490, 133 + 20 * k))[0] == 255 \
                and pyautogui.screenshot().getpixel((490, 133 + 20 * k))[1] == 255 \
                and pyautogui.screenshot().getpixel((490, 133 + 20 * k))[2] == 255):
                break
            pyautogui.click(490, 133 + 20 * k)
            k += 1
            time.sleep(2)
            pyautogui.click(1576, 216)
            time.sleep(0.5)
            cnt = 0
            for i in range(25):
                if np.array_equiv(pyautogui.screenshot().getpixel((1192, 453 + 20 * i)), [0, 0, 0]):
                    extract(i)
                    cnt += 1
                    print('#{} patient #{} visit #{} data'.format(p, k, cnt))
            img_old = pyautogui.screenshot().crop((600, 500, 1000, 700))
            pyautogui.press('down')
            img_new = pyautogui.screenshot().crop((600, 500, 1000, 700))
            diff = ImageChops.difference(img_old, img_new)
            while diff.getbbox():
                img_old = img_new
                extract(24)
                cnt += 1
                print('#{} patient #{} visit #{} data'.format(p, k, cnt))
                pyautogui.press('down')
                img_new = pyautogui.screenshot().crop((600, 500, 1000, 700))
                diff = ImageChops.difference(img_old, img_new)
        pyautogui.click(475, 1000)
        pyautogui.press('down')
