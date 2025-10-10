import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
from core import EMGLabelingCore
import os

class EMGLabelingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('EMG 라벨링 툴')
        self.core = EMGLabelingCore()
        
        # 윈도우 닫기 버튼(X) 클릭 시 처리
        self.root.protocol("WM_DELETE_WINDOW", self.quit_application)
        
        self.create_menu()
        self.setup_key_bindings()
        self.setup_ui()
        self.select_folder()

    def create_menu(self):
        """메뉴바를 생성합니다."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="폴더 변경", command=self.change_folder)
        file_menu.add_command(label="CSV 파일 불러오기", command=self.load_csv_file)
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.quit_application)
        
        # 도구 메뉴
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도구", menu=tools_menu)
        tools_menu.add_command(label="라벨링 통계", command=self.show_stats)
        
        # 도움말 메뉴
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="도움말", menu=help_menu)
        help_menu.add_command(label="단축키", command=self.show_shortcuts)
        help_menu.add_command(label="정보", command=self.show_about)

    def setup_key_bindings(self):
        """키보드 이벤트 바인딩을 설정합니다."""
        self.root.bind_all('<Key>', self.key_handler)
        self.root.bind_all('<Left>', lambda e: self.go_back())
        self.root.bind_all('<Right>', lambda e: self.go_next())
        self.root.bind_all('<bracketleft>', lambda e: self.change_scale(-1))  # [ 키
        self.root.bind_all('<bracketright>', lambda e: self.change_scale(1))  # ] 키
        self.root.bind_all('v', lambda e: self.save_label('Volition'))
        self.root.bind_all('n', lambda e: self.save_label('Non-volition'))

    def setup_ui(self):
        """UI 요소들을 초기화합니다."""
        # 스케일과 클러스터 정보를 표시할 레이블
        self.lbl_info = tk.Label(self.root, text='', font=('Arial', 12))
        self.lbl_info.pack(pady=5)

        # 이미지와 라벨을 담을 프레임
        image_frame = tk.Frame(self.root)
        image_frame.pack(pady=5)
        
        self.lbl_img = tk.Label(image_frame)
        self.lbl_img.pack()
        
        # 현재 라벨 표시 레이블
        self.lbl_current_label = tk.Label(image_frame, text='', font=('Arial', 12, 'bold'), fg='red')
        self.lbl_current_label.pack(pady=5)

        # 스케일 전환 버튼을 위한 프레임
        scale_frame = tk.Frame(self.root)
        scale_frame.pack(pady=5)
        
        btn_prev_scale = tk.Button(scale_frame, text='◀ 이전 스케일', width=15, 
                                 command=lambda: self.change_scale(-1))
        btn_next_scale = tk.Button(scale_frame, text='다음 스케일 ▶', width=15, 
                                 command=lambda: self.change_scale(1))
        
        btn_prev_scale.grid(row=0, column=0, padx=5)
        btn_next_scale.grid(row=0, column=1, padx=5)

        # 이미지 이동 버튼을 위한 프레임
        nav_frame = tk.Frame(self.root)
        nav_frame.pack(pady=5)

        btn_prev_img = tk.Button(nav_frame, text='이전 이미지 (←)', width=15, 
                               command=self.go_back)
        btn_next_img = tk.Button(nav_frame, text='다음 이미지 (→)', width=15, 
                               command=self.go_next)
        
        btn_prev_img.grid(row=0, column=0, padx=5)
        btn_next_img.grid(row=0, column=1, padx=5)

        # 라벨 버튼을 위한 프레임
        label_frame = tk.Frame(self.root)
        label_frame.pack(pady=5)

        btn_vol = tk.Button(label_frame, text='Volition (v)', width=20, 
                          command=lambda: self.save_label('Volition'))
        btn_non_vol = tk.Button(label_frame, text='Non-volition (n)', width=20, 
                              command=lambda: self.save_label('Non-volition'))

        btn_vol.grid(row=0, column=0, padx=5)
        btn_non_vol.grid(row=0, column=1, padx=5)

        # 인덱스 초기화
        self.index = 0
        self.images = []

    def select_folder(self):
        """이미지 폴더를 선택하고 초기화합니다."""
        folder = filedialog.askdirectory(title='이미지 폴더 선택')
        if folder:
            if not self.core.set_image_folder(folder):
                messagebox.showerror("오류", "선택한 폴더에 scale_ 폴더가 없거나 이미지 파일이 없습니다.")
                self.root.quit()
                return

            # 기본 CSV 파일 경로 설정
            default_csv = os.path.join(os.getcwd(), 'labeled_results.csv')
            
            # CSV 파일이 없을 때만 초기화
            if not os.path.exists(default_csv):
                self.core.initialize_csv()
            else:
                self.core.LABEL_FILE = default_csv
                self.core.load_existing_labels()
                self.core.update_csv_with_new_images()
            
            self.load_images()
            self.load_image()
            self.update_info_label()
        else:
            messagebox.showerror("오류", "폴더를 선택하지 않았습니다. 프로그램을 종료합니다.")
            self.root.quit()

    def change_folder(self):
        """새로운 이미지 폴더를 선택합니다."""
        folder = filedialog.askdirectory(title='새 이미지 폴더 선택')
        if folder:
            if self.core.set_image_folder(folder):
                self.index = 0
                self.load_images()
                self.load_image()
                self.update_info_label()
            else:
                messagebox.showerror("오류", "선택한 폴더에 scale_ 폴더가 없거나 이미지 파일이 없습니다.")

    def load_images(self):
        """현재 스케일의 이미지 목록을 로드합니다."""
        self.images = self.core.get_images_in_current_scale()
        if not self.images:
            messagebox.showerror("오류", "현재 스케일 폴더에 이미지가 없습니다.")
            self.root.quit()

    def load_image(self):
        """현재 이미지를 로드하고 표시합니다."""
        if self.index >= len(self.images):
            self.lbl_img.config(image='')
            self.lbl_info.config(text='라벨링 완료!')
            self.lbl_current_label.config(text='')
            return
            
        file = self.images[self.index]
        cluster = self.core.get_cluster(file)
        self.core.current_cluster = cluster
        
        current_scale_folder = self.core.get_current_scale_folder()
        path = os.path.join(current_scale_folder, file)
        img = Image.open(path).convert("RGB").resize((512, 512))

        photo = ImageTk.PhotoImage(img)
        self.lbl_img.config(image=photo)
        self.lbl_img.image = photo
        
        self.update_info_label()
        
        # 현재 라벨 표시 업데이트 (확장자 제외)
        label = self.core.labels.get(file)
        if label is not None:
            self.lbl_current_label.config(text=f"현재 라벨: {label}")
        else:
            self.lbl_current_label.config(text=f"현재 라벨: 미지정")

    def update_info_label(self):
        """정보 레이블을 업데이트합니다."""
        scale = self.core.get_current_scale()
        if scale and self.core.current_cluster:
            self.lbl_info.config(text=f"Scale: {scale} | Cluster: {self.core.current_cluster}")

    def change_scale(self, direction):
        """스케일을 변경합니다."""
        if self.index < len(self.images):
            current_filename = self.images[self.index]  # 현재 이미지 파일명 저장
        else:
            current_filename = None
            
        if self.core.change_scale(direction):
            self.load_images()
            
            # 이전 이미지 파일명이 있다면 같은 파일명을 찾아 그 위치로 이동
            if current_filename and current_filename in self.images:
                self.index = self.images.index(current_filename)
            else:
                self.index = 0
                
            self.load_image()

    def go_back(self):
        """이전 이미지로 이동합니다."""
        if self.index > 0:
            self.index -= 1
            self.load_image()

    def go_next(self):
        """다음 이미지로 이동합니다."""
        if self.index < len(self.images) - 1:
            self.index += 1
            self.load_image()

    def save_label(self, label):
        """현재 이미지에 라벨을 저장하고 다음 이미지로 이동합니다."""
        if self.index < len(self.images):
            filename = self.images[self.index]
            if self.core.save_label(filename, label):
                self.load_image()
                # 다음 이미지가 있으면 자동으로 이동
                if self.index < len(self.images) - 1:
                    self.index += 1
                    self.load_image()
                else:
                    # 마지막 이미지인 경우 완료 메시지 표시
                    messagebox.showinfo("완료", "현재 스케일의 모든 이미지 라벨링이 완료되었습니다.")

    def show_stats(self):
        """라벨링 통계를 표시합니다."""
        labeled_count, total_count, stats = self.core.get_stats()
        
        stat_text = f"라벨링 진행 상황: {labeled_count}/{total_count} ({labeled_count/total_count*100:.1f}%)\n\n"
        stat_text += "라벨별 통계:\n"
        for label in ['Volition', 'Non-volition']:
            count = stats.get(label, 0)
            percentage = count/labeled_count*100 if labeled_count > 0 else 0
            stat_text += f"{label}: {count}개 ({percentage:.1f}%)\n"
        
        messagebox.showinfo("라벨링 통계", stat_text)

    def show_shortcuts(self):
        """단축키 도움말을 표시합니다."""
        shortcuts = """
단축키 목록:
← : 이전 이미지
→ : 다음 이미지
[ : 이전 스케일
] : 다음 스케일
v : Volition 라벨
n : Non-volition 라벨
"""
        messagebox.showinfo("단축키", shortcuts)

    def show_about(self):
        """프로그램 정보를 표시합니다."""
        about_text = """
EMG 라벨링 툴 v2.0

이 프로그램은 EMG 이미지 데이터셋을 라벨링하기 위한 도구입니다.
각 이미지에 대해 Volition 또는 Non-volition 라벨을 지정할 수 있습니다.

© 2024 EMG 라벨링 툴
"""
        messagebox.showinfo("정보", about_text)

    def key_handler(self, event):
        """키보드 이벤트를 처리합니다."""
        try:
            if event.keysym == 'bracketleft':  # [ 키
                self.change_scale(-1)
            elif event.keysym == 'bracketright':  # ] 키
                self.change_scale(1)
            elif event.char == 'v':
                self.save_label('Volition')
            elif event.char == 'n':
                self.save_label('Non-volition')
            elif event.keysym == 'Left':
                self.go_back()
            elif event.keysym == 'Right':
                self.go_next()
        except Exception as e:
            print(f"키 이벤트 처리 중 오류 발생: {e}")

    def load_csv_file(self):
        """새로운 CSV 파일을 불러옵니다."""
        if messagebox.askyesno("CSV 파일 불러오기", 
                             "현재 라벨 데이터가 모두 삭제되고 새로운 CSV 파일의 데이터로 대체됩니다.\n계속하시겠습니까?"):
            file_path = filedialog.askopenfilename(
                title='CSV 파일 선택',
                filetypes=[('CSV 파일', '*.csv')]
            )
            if file_path:
                success, message = self.core.load_csv_file(file_path)
                if success:
                    messagebox.showinfo("성공", message)
                    # 현재 이미지 목록과 표시 업데이트
                    self.load_images()
                    self.load_image()
                else:
                    messagebox.showerror("오류", message)

    def quit_application(self):
        """프로그램을 종료하기 전에 항상 CSV 저장 위치를 지정합니다."""
        file_path = filedialog.asksaveasfilename(
            title='CSV 파일 저장 위치 선택 (종료 전 필수)',
            defaultextension='.csv',
            filetypes=[('CSV 파일', '*.csv')],
            initialfile='labeled_results.csv'
        )
        if file_path:
            success, message = self.core.finalize_csv(file_path)
            if success:
                messagebox.showinfo("저장 완료", message)
                self.root.quit()
            else:
                messagebox.showerror("오류", message)
                if not messagebox.askretrycancel("재시도", "CSV 파일 저장을 다시 시도하시겠습니까?"):
                    self.root.quit()
                else:
                    self.quit_application()
        else:
            if messagebox.askyesno("저장 취소", "CSV 파일을 저장하지 않고 종료하시겠습니까?"):
                self.root.quit()
            else:
                self.quit_application() 