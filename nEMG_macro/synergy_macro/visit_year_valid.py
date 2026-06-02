import os
import shutil
from pathlib import Path
from datetime import datetime

def migrate_and_cleanup_files(root_path, dry_run=True):
    root_dir = Path(root_path)
    count_moved = 0
    count_deleted = 0

    # 1. 루트 폴더 내의 모든 연도별 하위 폴더 순회
    for year_folder in root_dir.iterdir():
        if not year_folder.is_dir() or not year_folder.name.isdigit():
            continue
        
        current_folder_year = year_folder.name
        
        # 2. 해당 폴더 내의 txt 파일 순회
        # 리스트로 변환해두어야 파일 이동/삭제 시 반복문 꼬임을 방지할 수 있습니다.
        for file_path in list(year_folder.glob("*.txt")):
            try:
                # 파일명에서 실제 연도 추출 (사용자 로직)
                parts = file_path.name.split(" - ")
                if len(parts) < 2: continue
                
                date_str = parts[1]
                date_str_eng = date_str.replace("오전", "AM").replace("오후", "PM")
                test_time = datetime.strptime(date_str_eng, "%Y-%m-%d %p %I_%M_%S")
                actual_year = str(test_time.year)

                # 3. 연도 불일치 시 처리 로직
                if current_folder_year != actual_year:
                    target_folder = root_dir / actual_year
                    target_file_path = target_folder / file_path.name

                    # 대상 연도 폴더가 없으면 생성
                    if not target_folder.exists() and not dry_run:
                        target_folder.mkdir(parents=True)

                    # 중복 확인
                    if target_file_path.exists():
                        # 이미 대상 폴더에 파일이 있으면 현재 파일(잘못된 위치) 삭제
                        print(f"[중복 삭제] {file_path.name} (이미 {actual_year} 폴더에 존재)")
                        if not dry_run:
                            file_path.unlink()
                        count_deleted += 1
                    else:
                        # 대상 폴더에 파일이 없으면 이동
                        print(f"[파일 이동] {file_path.name}: {current_folder_year} -> {actual_year}")
                        if not dry_run:
                            shutil.move(str(file_path), str(target_file_path))
                        count_moved += 1

            except Exception as e:
                print(f"[에러] {file_path.name} 처리 중 오류: {e}")

    print(f"\n{'[Dry Run 결과]' if dry_run else '[실행 완료]'}")
    print(f"- 이동된 파일: {count_moved}개")
    print(f"- 중복 삭제된 파일: {count_deleted}개")


if __name__ == '__main__':
    target_root = "/home/coder/workspace/nas1/EMG/EMG_raw_txt/site1"
    migrate_and_cleanup_files(target_root, dry_run=True)        # 실제 삭제 없는 시뮬레이션

    # 2. 결과 확인 후 문제가 없다면 아래 주석 해제 후 실행
    # migrate_and_cleanup_files(target_root, dry_run=False)