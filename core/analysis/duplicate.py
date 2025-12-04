import os

import imagehash
from PIL import Image
from PySide6.QtCore import QThread, Signal


class DuplicateFinderThread(QThread):
    progress = Signal(int, int)  # current, total
    finished = Signal(dict)  # results {hash: [img_ids]}
    error = Signal(str)

    def __init__(self, images_dict, img_root):
        super().__init__()
        self.images_dict = images_dict
        self.img_root = img_root
        self.is_running = True

    def run(self):
        try:
            hash_dict = {}
            total = len(self.images_dict)

            for idx, (img_id, img_info) in enumerate(self.images_dict.items()):
                if not self.is_running:
                    break

                file_path = os.path.join(self.img_root, img_info["file_name"])
                if not os.path.exists(file_path):
                    continue

                try:
                    with Image.open(file_path) as img:
                        # Average Hash가 빠르고 일반적인 중복 검출에 적합
                        # phash(Perceptual Hash)가 변형에 강함
                        img_hash = str(imagehash.phash(img))

                        if img_hash in hash_dict:
                            hash_dict[img_hash].append(img_id)
                        else:
                            hash_dict[img_hash] = [img_id]

                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

                if idx % 10 == 0:
                    self.progress.emit(idx + 1, total)

            # 중복된 것만 필터링 (2개 이상인 그룹)
            duplicates = {k: v for k, v in hash_dict.items() if len(v) > 1}
            self.finished.emit(duplicates)

        except Exception as e:
            self.error.emit(str(e))

    def stop(self):
        self.is_running = False

    @staticmethod
    def convert_to_groups(duplicates_dict):
        """Convert duplicates dictionary to list of sets.

        Args:
            duplicates_dict: Dictionary {hash: [img_ids]} from finished signal.

        Returns:
            List of sets, each containing duplicate image IDs.
        """
        return [
            set(img_ids) for img_ids in duplicates_dict.values() if len(img_ids) > 1
        ]
