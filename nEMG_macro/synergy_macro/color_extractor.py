import pyautogui 
import keyboard

while True:
    key = keyboard.read_event(suppress=True).name
    
    if key == 'q':
        pos = pyautogui.position()
        im = pyautogui.screenshot()
        print(f"좌표: {pos} / RGB 값: {im.getpixel(pos)}")
        print()
        
    elif key == 'e':
        try:
            coords = input("X,Y 좌표 입력 : ").split(',')
            x, y = map(int, coords)
            im = pyautogui.screenshot()
            print(f"좌표: ({x}, {y}) / RGB 값: {im.getpixel((x, y))}")
            print()
        except ValueError:
            print("올바른 숫자를 입력해주세요.")
            print()
        except Exception as e:
            print(f"오류 발생: {e}")
            print()