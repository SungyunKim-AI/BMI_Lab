CoordMode("Mouse", "Screen")
CoordMode("Pixel", "Screen")

Esc::ExitApp() ; ESC 키를 누르면 강제 종료
start_macro()

start_macro() {
    Sleep 3000
    while true
    {
        Click 350, 280 ; 제일 첫 환자 클릭 및 시작
        j := 0
        while true ; Visit ID 데이터 반복
        {
            Sleep 500
            visit_y := 730 + (20 * j)
            Click 350, visit_y ; Visit ID 데이터 있는지 확인
            Sleep 500
            color := PixelGetColor(350, visit_y, "RGB") ; 클릭했을 때 파란색이면 데이터 있는것
            ;MsgBox(color)
            if (color = 0x3399FF) ; 파란색:(0x3399FF)
            {
                Click 1450, 100 ; 리포트 버튼 클릭
                Sleep 2000
                Click 900, 435 ; Curve_Data 포맷클릭
                Sleep 500
                Click 925, 600 ; 확인 클릭
                Sleep 5000
                Click 300, 90 ; 리포트 파일 저장
                Sleep 1000
                Click 580, 585 ; 텍스트 입력
                SendText "curve_data"
                Sleep 1000
                ; Send "{Enter}"
                Click 1000, 720
                Sleep 1000
                ; Send "!{F4}" ; 워드 파일 닫기기
                Click 1665, 90
                Sleep 1000
            }
            else
            {
                break
            }
            j := j + 1
        }
        Send "{Down}" ; 아랫방향키 입력
        Sleep 500
        side_bar_color := PixelGetColor(1160, 643, "RGB")
        if (side_bar_color = 0x696969) ; 이동할 환자가 없다고 판단, 반복 종료
            break
    }
}

