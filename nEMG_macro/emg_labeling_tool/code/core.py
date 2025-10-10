import os
import csv
import tempfile
import shutil

class EMGLabelingCore:
    def __init__(self):
        self.labels = {}  # 파일이름: 라벨
        self.image_folder = None
        self.scale_folders = []  # 스케일 폴더 목록
        self.current_scale_index = 0  # 현재 선택된 스케일 폴더 인덱스
        self.all_images = set()  # 모든 스케일의 이미지 파일명 집합
        self.current_cluster = None
        self.LABEL_FILE = None  # 최종 저장될 CSV 파일 경로
        self.temp_file = None  # 임시 파일 경로
        self.IMAGE_EXT = ('.png', '.jpg', '.jpeg')
        self._create_temp_file()

    def _create_temp_file(self):
        """임시 파일을 생성합니다."""
        if self.temp_file and os.path.exists(self.temp_file):
            os.remove(self.temp_file)
        fd, self.temp_file = tempfile.mkstemp(suffix='.csv')
        os.close(fd)

    def _remove_extension(self, filename):
        """파일명에서 확장자를 제거합니다."""
        return os.path.splitext(filename)[0]

    def find_scale_folders(self):
        """현재 선택된 폴더의 하위 폴더 중에서 스케일 폴더들을 찾습니다."""
        if not self.image_folder:
            return []
        
        scale_folders = []
        for item in os.listdir(self.image_folder):
            full_path = os.path.join(self.image_folder, item)
            if os.path.isdir(full_path) and item.startswith('scale_'):
                try:
                    scale_num = int(item.split('_')[1])
                    scale_folders.append((scale_num, item))
                except (ValueError, IndexError):
                    continue
        
        scale_folders.sort(key=lambda x: x[0])
        return [folder[1] for folder in scale_folders]

    def collect_all_images(self):
        """모든 스케일 폴더의 이미지 파일들을 수집합니다."""
        all_images = set()
        for scale_folder in self.scale_folders:
            scale_path = os.path.join(self.image_folder, scale_folder)
            images = [f for f in os.listdir(scale_path) if f.endswith(self.IMAGE_EXT)]
            all_images.update(images)
        return sorted(list(all_images))

    def get_cluster(self, filename):
        """파일명에서 클러스터 정보를 추출합니다."""
        try:
            folders = os.path.normpath(self.image_folder).split(os.sep)
            cluster_folder = [folder for folder in folders if 'cluster' in folder]
            if cluster_folder:
                cluster = cluster_folder[0].split('_')[1]
                return cluster
            else:
                return 'unknown'
        except Exception as e:
            print(f"클러스터 정보 추출 중 오류 발생: {e}")
            return 'unknown'

    def initialize_csv(self):
        """모든 이미지 파일에 대한 CSV 파일을 초기화합니다."""
        with open(self.temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'cluster', 'label'])
            for filename in self.all_images:
                filename_without_ext = self._remove_extension(filename)
                writer.writerow([filename_without_ext, self.get_cluster(filename), None])

    def load_existing_labels(self):
        """기존 CSV 파일에서 라벨 정보를 로드합니다."""
        self.labels = {}
        if os.path.exists(self.LABEL_FILE):
            with open(self.LABEL_FILE, 'r') as f:
                reader = csv.reader(f)
                next(reader, None)  # 헤더 건너뛰기
                for row in reader:
                    if len(row) >= 3:
                        filename_without_ext, cluster, label = row
                        # 원본 파일명으로 매핑 (확장자가 있는 파일명 찾기)
                        for ext in self.IMAGE_EXT:
                            full_filename = filename_without_ext + ext
                            if full_filename in self.all_images:
                                self.labels[full_filename] = None if label == '' else label
                                break
                        if filename_without_ext not in [self._remove_extension(f) for f in self.all_images]:
                            self.all_images.add(filename_without_ext + self.IMAGE_EXT[0])  # 기본 확장자 사용
            self.all_images = sorted(list(self.all_images))
            # 임시 파일에 현재 데이터 복사
            self.save_to_csv()

    def update_csv_with_new_images(self):
        """기존 CSV 파일에 새로운 이미지 파일들을 추가합니다."""
        existing_files = set()
        new_rows = []
        
        if os.path.exists(self.LABEL_FILE):
            with open(self.LABEL_FILE, 'r', newline='') as f:
                reader = csv.reader(f)
                header = next(reader)
                for row in reader:
                    if len(row) >= 3:
                        filename_without_ext = row[0]
                        existing_files.add(filename_without_ext)
                        label = None if row[2] == '' else row[2]
                        new_rows.append([filename_without_ext, row[1], label])
        
        for filename in self.all_images:
            filename_without_ext = self._remove_extension(filename)
            if filename_without_ext not in existing_files:
                new_rows.append([filename_without_ext, self.get_cluster(filename), None])
        
        new_rows.sort(key=lambda x: x[0])
        
        with open(self.temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'cluster', 'label'])
            writer.writerows(new_rows)

    def save_to_csv(self):
        """현재 라벨 정보를 임시 CSV 파일에 저장합니다."""
        with open(self.temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'cluster', 'label'])
            for filename in self.all_images:
                filename_without_ext = self._remove_extension(filename)
                label = self.labels.get(filename)
                writer.writerow([filename_without_ext, self.get_cluster(filename), '' if label is None else label])

    def finalize_csv(self, target_path):
        """임시 파일을 최종 위치로 이동하고 임시 파일을 삭제합니다."""
        try:
            # 임시 파일을 최종 위치로 복사
            shutil.copy2(self.temp_file, target_path)
            self.LABEL_FILE = target_path
            # 임시 파일 삭제
            if os.path.exists(self.temp_file):
                os.remove(self.temp_file)
            self.temp_file = None
            return True, "CSV 파일이 성공적으로 저장되었습니다."
        except Exception as e:
            return False, f"CSV 파일 저장 중 오류 발생: {str(e)}"

    def get_stats(self):
        """라벨링 통계를 계산합니다."""
        labeled_values = [v for v in self.labels.values() if v is not None]
        labeled_count = len(labeled_values)
        total_count = len(self.all_images)
        stats = {x: labeled_values.count(x) for x in ['Volition', 'Non-volition']}
        return labeled_count, total_count, stats

    def set_image_folder(self, folder):
        """이미지 폴더를 설정하고 초기화합니다."""
        self.image_folder = folder
        self.scale_folders = self.find_scale_folders()
        self.all_images = self.collect_all_images()
        self.current_scale_index = len(self.scale_folders) // 2
        return bool(self.scale_folders and self.all_images)

    def get_current_scale_folder(self):
        """현재 스케일 폴더의 경로를 반환합니다."""
        if not self.scale_folders:
            return None
        return os.path.join(self.image_folder, self.scale_folders[self.current_scale_index])

    def get_current_scale(self):
        """현재 스케일 값을 반환합니다."""
        if not self.scale_folders:
            return None
        return self.scale_folders[self.current_scale_index].split('_')[1]

    def change_scale(self, direction):
        """스케일 폴더를 변경합니다."""
        if not self.scale_folders:
            return False
        self.current_scale_index = (self.current_scale_index + direction) % len(self.scale_folders)
        return True

    def get_images_in_current_scale(self):
        """현재 스케일 폴더의 이미지 목록을 반환합니다."""
        if not self.scale_folders:
            return []
        current_scale_folder = self.get_current_scale_folder()
        return sorted([f for f in os.listdir(current_scale_folder) if f.endswith(self.IMAGE_EXT)])

    def save_label(self, filename, label):
        """이미지에 라벨을 저장합니다."""
        self.labels[filename] = label
        self.save_to_csv()
        return True

    def set_csv_path(self, path):
        """CSV 파일의 저장 경로를 설정합니다."""
        if path:
            self.LABEL_FILE = path
            # 기존 라벨 데이터가 있다면 새 위치로 복사
            if os.path.exists('labeled_results.csv'):
                self.save_to_csv()
            return True
        return False

    def load_csv_file(self, file_path):
        """새로운 CSV 파일을 불러와서 라벨 데이터를 교체합니다."""
        if not os.path.exists(file_path):
            return False, "파일이 존재하지 않습니다."
            
        try:
            new_labels = {}
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header or len(header) < 3 or header[0] != 'filename' or header[1] != 'cluster' or header[2] != 'label':
                    return False, "잘못된 CSV 파일 형식입니다."
                
                for row in reader:
                    if len(row) >= 3:
                        filename, cluster, label = row
                        new_labels[filename] = None if label == '' else label
                        if filename not in self.all_images:
                            self.all_images.add(filename)
            
            self.labels = new_labels
            self.all_images = sorted(list(self.all_images))
            self.LABEL_FILE = file_path
            # 임시 파일에 현재 데이터 복사
            self.save_to_csv()
            return True, "CSV 파일을 성공적으로 불러왔습니다."
            
        except Exception as e:
            return False, f"CSV 파일 로드 중 오류 발생: {str(e)}"

    def __del__(self):
        """객체가 소멸될 때 임시 파일을 정리합니다."""
        if self.temp_file and os.path.exists(self.temp_file):
            try:
                os.remove(self.temp_file)
            except:
                pass 