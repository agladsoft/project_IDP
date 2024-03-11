import re
import sys
import cv2
import csv
import glob
import yaml
import math
import magic
import shutil
import easyocr
import requests
import warnings
import contextlib
import pytesseract
import pandas as pd
import scipy.ndimage
from re import Match
from PIL import Image
from __init__ import *
from numpy import ndarray
from fuzzywuzzy import fuzz
from PyPDF2 import PdfFileReader
from multiprocessing import Pool
from itertools import combinations
from pdf2image import convert_from_path
from configuration import Configuration
from collections import namedtuple, defaultdict
from validations_post_processing import DataValidator
from typing import List, Optional, Tuple, Union, Sequence


logger: getLogger = get_logger("logging")


class HandleMultiPagesPDF:
    """
    Класс для работы с многостраничным pdf (разделение страниц на jpg и запуск последующих классов).
    """
    def __init__(self, file: str, dir_cache: str):
        self.file: str = file
        self.dir_cache: str = dir_cache

    def get_count_pages_from_file(self) -> int:
        """
        Получить количество файлов в pdf.
        :param:
        :return:
        """
        with open(self.file, 'rb') as fl:
            page_num: int = PdfFileReader(fl).getNumPages()
            logger.info(f"Количество файлов pdf в многостраничном файле: {page_num}")
            with contextlib.suppress(Exception):
                requests.post(f"http://{IP_ADDRESS_FOR_SEND_RESULT}:5000/get_logs", json={
                    "logs": "Получение количества файлов pdf в многостраничном файле",
                    "value": page_num
                })
            return page_num

    def split_file(self) -> None:
        """
        Разделить многостраничный pdf на множество jpg файлов.
        :return:
        """
        images: List[Image] = convert_from_path(self.file)
        for i in range(len(images)):
            images[i].save(f'{self.dir_cache}/{os.path.basename(self.file)}_{str(i)}.jpg', 'JPEG')

    def get_count_split_pages(self) -> None:
        """
        Получить количество разделенных файлов.
        :return:
        """
        path, dirs, files = next(os.walk(self.dir_cache))
        file_count: int = len(files)
        logger.info(f"Количество разбитых файлов pdf в кэше: {file_count}")

    def start_file_processing(self) -> None:
        """
        Запуск разделенных файлов через мультипроцессинг.
        :return:
        """
        procs: list = []
        dir_main: str = os.path.dirname(self.file)
        with Pool(processes=WORKER_COUNT) as pool:
            for file in glob.glob(f"{self.dir_cache}/*.jpg"):
                proc = pool.apply_async(self.handle_pages, (file, dir_main, f"{dir_main}/csv", f"{dir_main}/json",
                                                            f"{dir_main}/classification"))
                procs.append(proc)
            [proc.get() for proc in procs]

    def handle_pages(self, file: str, dir_main: str, dir_csv: str, dir_json: str, classification: str) -> None:
        """
        Обработка каждой страницы (поворот изображения, классификация).
        :param file: Разделенный файл jpg.
        :param dir_main: Основная директория для загрузки файла.
        :param dir_csv: Директория для сохранения файлов в формате csv.
        :param dir_json: Директория для сохранения файлов в формате json.
        :param classification: Тип классификации.
        :return:
        """
        classification_yml: str = f"{os.path.dirname(os.path.dirname(self.file))}/configs"
        classification_name: str = HandleJPG(
            file, dir_main, classification, f"{classification_yml}/classification"
        ).main()
        RecognizeTable(
            f"{classification}/{classification_name}/{os.path.basename(file)}",
            dir_json,
            dir_csv,
            f"{classification_yml}/{classification_name}/{classification_name}.yml",
            f"{classification_yml}/{classification_name}/{classification_name}.py"
        ).main()

    def main(self) -> Optional[str]:
        """
        Основной метод, который запускает код.
        :return:
        """
        mime_type: str = magic.from_file(self.file, mime=True)
        if mime_type == "application/pdf":
            self.get_count_pages_from_file()
            self.split_file()
            self.get_count_split_pages()
        elif mime_type == "image/jpeg":
            shutil.copy(self.file, f'{self.dir_cache}/{os.path.basename(self.file)}')
        else:
            message: str = f"The file format is not supported {os.path.basename(self.file)}"
            logger.error(message)
            return message
        self.start_file_processing()


class HandleJPG:
    """
    Класс для обработки jpg (поворот изображения, классификация).
    """
    def __init__(self, input_file: str, dir_main: str, classification: str, configs: str):
        self.input_file: str = input_file
        self.dir_main: str = dir_main
        self.classification: str = classification
        self.configs: str = configs

    def rotate(self, image: ndarray, angle: int, is_right_angle: bool, background: Optional[tuple] = None) -> ndarray:
        """
        Поворот изображения на 90, 180 или 270 градусов.
        :param image: Исходное изображение в виде матрицы.
        :param angle: Угол, на который нужно повернуть.
        :param background: Оттенок серого цвета.
        :param is_right_angle: Прямой ли этот угол
        :return: Матрица перевернутого изображения.
        """
        old_width, old_height = image.shape[:2]
        logger.info(f'Rotate: {angle}, Is_right_angle: {is_right_angle}, Filename: {os.path.basename(self.input_file)}')
        if is_right_angle:
            angle_radian: float = math.radians(angle)
            width: float = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
            height: float = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)
            image_center: Tuple[float, float] = tuple(np.array(image.shape[1::-1]) / 2)
            rot_mat: ndarray = cv2.getRotationMatrix2D(image_center, angle, 1.0)
            rot_mat[1, 2] += (width - old_width) / 2
            rot_mat[0, 2] += (height - old_height) / 2
            return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)
        center: Tuple[int, int] = (old_height // 2, old_width // 2)
        M: ndarray = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (old_height, old_width), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    def turn_img(self):
        """
        Поворот изображения на прямой угол (90, 180, 270).
        :return:
        """
        if not os.path.exists(self.classification):
            os.makedirs(os.path.join(self.classification, "line"))
            os.makedirs(os.path.join(self.classification, "port"))
            os.makedirs(os.path.join(self.classification, "contract"))
            os.makedirs(os.path.join(self.classification, "unknown"))

        image: ndarray = cv2.imread(self.input_file)
        rotate_img: str = pytesseract.image_to_osd(image, config='--psm 0 -c min_characters_to_try=5')
        angle_rotated_image: int = int(re.search(r'(?<=Orientation in degrees: )\d+', rotate_img)[0])
        rotated: ndarray = self.rotate(image, angle_rotated_image, is_right_angle=True, background=(0, 0, 0))
        self.save_file(angle_rotated_image, rotated, "Поворот изображений на прямой угол")

    def correct_skew(self, delta: int, limit: int) -> None:
        """
        Поворот изображения на маленький угол.
        :param delta: Шаг для нахождения нужного угла.
        :param limit: Максимальный допустимый угол для поворота.
        :return:
        """
        image: ndarray = cv2.imread(self.input_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh: Union[ndarray, cv2.UMat] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        X_Y: namedtuple = namedtuple("X_Y", "x y")
        dict_angle_and_score: dict = {0: X_Y(0, self._determine_score(thresh, 0))}
        best_angle: Union[int, float] = 0
        for angle in range(1, limit, delta):
            dict_angle_and_score[angle] = X_Y(angle, self._determine_score(thresh, angle))
            dict_angle_and_score[-angle] = X_Y(-angle, self._determine_score(thresh, -angle))
            sorted_x_y: list = sorted(dict_angle_and_score.values(), key=lambda xy: xy.y)
            max_value: X_Y = sorted_x_y[-1]
            min_value: X_Y = sorted_x_y[0]
            if max_value.y > min_value.y * 10:
                left: X_Y = dict_angle_and_score.get(max_value.x - 1)
                right: X_Y = dict_angle_and_score.get(max_value.x + 1)
                if left and right:
                    best_angle = self._golden_ratio(left.x, right.x, 0.1, thresh)
                    best_score: float = self._determine_score(thresh, best_angle)
                    if best_score > min_value.y * 100:
                        break
                    else:
                        del dict_angle_and_score[max_value.x]

        corrected: ndarray = self.rotate(image, best_angle, is_right_angle=False)
        self.save_file(best_angle, corrected, "Выравнивание изображений под маленьким углом")

    def save_file(self, angle: Union[int, float], ndarray_image: ndarray, message_to_send: str) -> None:
        """
        Сохраняем перевернутое изображение.
        :param angle: Угол, на который повернули.
        :param ndarray_image: Изображение в виде матрицы.
        :param message_to_send: Сообщение для отправки и логирования.
        :return:
        """
        file_name: str = os.path.basename(self.input_file)
        cv2.imwrite(f'{os.path.dirname(self.input_file)}/{file_name}', ndarray_image)
        with contextlib.suppress(Exception):
            requests.post(f"http://{IP_ADDRESS_FOR_SEND_RESULT}:5000/get_logs", json={
                "logs": message_to_send,
                "value": angle
            })

    @staticmethod
    def _determine_score(arr: Union[ndarray, cv2.UMat], angle: Union[int, float]) -> float:
        """
        Определяем наилучший результат для угла.
        :param arr: Изображение в виде матрицы.
        :param angle: Угол, на который нужно повернуть.
        :return:
        """
        data: ndarray = scipy.ndimage.rotate(arr, angle, reshape=False, order=0)
        histogram: ndarray = np.sum(data, axis=1, dtype=float)
        score: np.array_api.float64 = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
        return score // 100000000

    def _golden_ratio(self, left: int, right: int, delta: float, thresh: Union[ndarray, cv2.UMat]) -> Union[int, float]:
        """
        Определяем лучший угол для поворота (по золотому сечению). Вот ссылка для ознакомления
        https://en.wikipedia.org/wiki/Golden_ratio
        :param left: Минимальный диапазон для нахождения угла.
        :param right: Максимальный диапазон для нахождения угла.
        :param delta: Минимальный диапазон для нахождения угла.
        :param thresh: Изображение в виде матрицы.
        :return: Наилучший найденный угол для поворота.
        """
        res_phi: float = 2 - (1 + math.sqrt(5)) / 2
        x1, x2 = left + res_phi * (right - left), right - res_phi * (right - left)
        f1, f2 = self._determine_score(thresh, x1), self._determine_score(thresh, x2)
        scores: List[float] = []
        angles: List[float] = []
        while abs(right - left) > delta:
            if f1 < f2:
                left, x1, f1 = x1, x2, f2
                x2: float = right - res_phi * (right - left)
                f2: float = self._determine_score(thresh, x2)
                scores.append(f2)
                angles.append(x2)
            else:
                right, x2, f2 = x2, x1, f1
                x1: float = left + res_phi * (right - left)
                f1: float = self._determine_score(thresh, x1)
                scores.append(f1)
                angles.append(x1)
        return (x1 + x2) / 2

    def move_file(self, file_name: str, yaml_file: dict, label_in_config: str) -> str:
        """
        Переместить файл в категорию классификаций.
        :param file_name: Наименование файла.
        :param yaml_file: Словарь с настройками конфиг файла.
        :param label_in_config: Наименование категории из конфиг файла.
        :return: Категория из конфиг файла.
        """
        directory: str = yaml_file[label_in_config]['folder']
        file: str = f"{directory}/{os.path.basename(file_name)}"
        os.remove(file) if os.path.isfile(file) else None
        if os.path.isfile(file_name):
            shutil.copy2(file_name, f'{os.path.dirname(self.dir_main)}/{directory}')
            os.remove(file_name)
        else:
            shutil.move(file_name, f'{os.path.dirname(self.dir_main)}/{directory}')
        return yaml_file[label_in_config]['name']

    def read_config_file(self, str_of_doc: str, file_name: str, predict: Optional[str] = None) -> str:
        """
        Прочитать конфиг файл и определить нужную категорию для файла.
        :param str_of_doc: Строки из прочитанного документа.
        :param file_name: Наименования файла.
        :param predict: Категория из конфиг файла.
        :return: Категория из конфиг файла.
        """
        with open(f"{self.configs}/classification.yml", "r") as stream:
            try:
                yaml_file: dict = yaml.safe_load(stream)
                for label_in_config in yaml_file:
                    if yaml_file[label_in_config]["key"].upper() in str_of_doc.upper():
                        predict = self.move_file(file_name, yaml_file, label_in_config)
                if not predict:
                    logger.info(f'Filename: {os.path.basename(self.input_file)}, text: {str_of_doc}')
                    predict = self.move_file(file_name, yaml_file, "unknown")
            except yaml.YAMLError as exc:
                print("Exception", exc)
        return predict

    def classification_img(self) -> str:
        """
        Классификация файла по совпадению слов.
        :return: Категория из конфиг файла.
        """
        image: Image = Image.open(self.input_file)
        ocr_df: pd.DataFrame = pytesseract.image_to_data(image, output_type='data.frame', lang='rus+eng')
        float_cols: pd.Index = ocr_df.select_dtypes('float').columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        words: list = list(ocr_df.text)
        str_of_doc: str = " ".join(words[:20])
        predict: str = self.read_config_file(str_of_doc, self.input_file)
        logger.info(f'Filename: {os.path.basename(self.input_file)}, Predict class: {predict}')
        with contextlib.suppress(Exception):
            requests.post(f"http://{IP_ADDRESS_FOR_SEND_RESULT}:5000/get_logs", json={
                "logs": "Классификация изображений",
                "value": predict
            })
        return predict

    def main(self) -> str:
        """
        Основной метод, который запускает код.
        :return: Категория из конфиг файла.
        """
        self.turn_img()
        self.correct_skew(delta=1, limit=60)
        return self.classification_img()


class RecognizeTable:
    """
    Класс для распознавания текста внутри таблицы.
    """

    label_list: list = []
    bit_not: Optional[ndarray] = None

    length_of_kernel: Optional[int] = None
    min_height_of_cell: Optional[int] = None
    min_width_of_cell: Optional[int] = None
    indent_x_text_of_cells: Optional[int] = None
    indent_y_text_of_cells: Optional[int] = None
    config_for_pytesseract: Optional[str] = None

    database: Optional[str] = None
    user: Optional[str] = None
    password: Optional[str] = None
    host: Optional[str] = None
    port: Optional[str] = None
    table: Optional[str] = None

    def __init__(self, file: str, output_directory: str, output_directory_csv: str, config_yaml_file,
                 scripts_validations: str):
        self.file: str = file
        self.output_directory: str = output_directory
        self.output_directory_csv: str = output_directory_csv
        self.config_yaml_file: str = config_yaml_file
        self.scripts_validations: str = scripts_validations

    @staticmethod
    def add_values_in_dict(ocr_json_label: dict, dict_text: dict, value: str) -> None:
        """
        Добавление значений в словарь.
        :param ocr_json_label: Словарь для добавления данных.
        :param dict_text: Словарь, где содержатся текучие данные.
        :param value: Значение из словаря.
        :return:
        """
        ocr_json_label["text"] = value
        ocr_json_label["xmin"] = dict_text[value][0]
        ocr_json_label["ymin"] = dict_text[value][1]
        ocr_json_label["xmax"] = dict_text[value][2]
        ocr_json_label["ymax"] = dict_text[value][3]
        ocr_json_label["score"] = dict_text[value][4]
        ocr_json_label["std"] = dict_text[value][5]

    def write_to_file(self, dir_txt: str, str_data: str) -> None:
        """
        Сохраняем данные в файл.
        :param dir_txt: Наименование папки.
        :param str_data: Данные.
        :return:
        """
        with open(f'{dir_txt}/{os.path.basename(f"{self.file}.txt")}', "w", encoding="utf-8") as f:
            f.write(str_data)
            logger.info(f'Write to {dir_txt}/{os.path.basename(f"{self.file}.text")}')

    def extracted_text(self) -> None:
        """
        Извлекаем текст изображения.
        :return:
        """
        img: ndarray = cv2.imread(self.file, 1)
        dir_txt: str = f"{os.path.dirname(self.output_directory)}/txt"
        if not os.path.exists(dir_txt):
            os.makedirs(dir_txt)
        noiseless_image_colored = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)
        cv2.imwrite("x.jpg", noiseless_image_colored)
        data: dict = pytesseract.image_to_string(noiseless_image_colored, output_type="dict", lang='rus+eng')
        self.write_to_file(dir_txt, data["text"])

    @staticmethod
    def find_score_and_text(data: dict, labels: bool) -> Tuple[dict, str, list]:
        """
        Находим оценку и текст распознавания.
        :param data: Словарь распознанных данных.
        :param labels: Находим ли лейбл.
        :return: Оценку и текст распознавания.
        """
        boxes: int = len(data['level'])
        list_i: list = []
        dict_text: dict = {}
        list_score: list = []
        text_ocr: str = ''
        x_min = y_min = x_max = y_max = 0
        for i in range(boxes):
            (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            if data["text"][i] and not list_i[-1]:
                text_ocr = str(data["text"][i]) if labels else " ".join(data["text"])
                x_min, y_min = x, y
                list_score.clear()
                list_score.append(data["conf"][i])
            if data["text"][i] and list_i[-1] and labels:
                text_ocr += " " + str(data["text"][i])
                list_score.append(data["conf"][i])
                x_max, y_max = x + w, y + h
            elif not data["text"][i]:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    dict_text[text_ocr] = (x_min, y_min, x_max, y_max, np.mean(list_score), np.std(list_score))
            list_i.append(data["text"][i])

        return dict_text, text_ocr, list_score

    def parse_labels_from_config_file(self, data: dict, yaml_file: dict) -> None:
        """
        Парсим лейбл из конфиг файла.
        :param data: Словарь распознанных данных.
        :param yaml_file: Данные конфиг файла.
        :return:
        """
        dict_text: dict = self.find_score_and_text(data, labels=True)[0]
        for value in list(dict_text.keys()):
            for len_label_in_config in range(len(yaml_file["labels"])):
                if yaml_file["labels"][len_label_in_config]["key"].upper() in value.upper():
                    ocr_json_label: dict = {"type": "text", "label": yaml_file["labels"][len_label_in_config]["label"]}
                    self.add_values_in_dict(ocr_json_label, dict_text, value)
                    data_validator: DataValidator = DataValidator()
                    postprocessing: str = data_validator.postprocessing(
                        yaml_file, yaml_file["labels"][len_label_in_config]["key"], len_label_in_config,
                        self.scripts_validations, value
                    )
                    ocr_json_label["text"] = postprocessing
                    validations: bool = data_validator.validations(
                        yaml_file, len_label_in_config, self.scripts_validations, postprocessing,
                        ocr_json_label["score"]
                    )
                    ocr_json_label["is_valid"] = validations
                    self.label_list.append(ocr_json_label)

    def read_config_file(self, data: dict) -> None:
        """
        Чтение конфиг файла.
        :param data: Словарь распознанных данных.
        :return:
        """
        with open(self.config_yaml_file, "r") as stream:
            try:
                yaml_file: dict = yaml.safe_load(stream)
                self.length_of_kernel, self.min_height_of_cell, self.min_width_of_cell, self.indent_x_text_of_cells, \
                    self.indent_y_text_of_cells, self.config_for_pytesseract = \
                    Configuration().config_of_tables(yaml_file)
                self.database, self.user, self.password, self.host, self.port, self.table = \
                    Configuration().config_of_database(yaml_file)
                self.parse_labels_from_config_file(data, yaml_file)
            except yaml.YAMLError as exc:
                print(exc)

    def process_image_and_lines(self, img: ndarray) -> ndarray:
        """
        Выполнение предобработки изображения и выделение линий.
        :param img: Исходное изображение в виде матрицы.
        :return: Изображение с выделенными линиями.
        """
        thresh, img_bin = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img_bin = 255 - img_bin
        kernel_len = np.array(img).shape[1] // self.length_of_kernel
        ver_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
        hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        image_1 = cv2.erode(img_bin, ver_kernel, iterations=3)
        vertical_lines = cv2.dilate(image_1, ver_kernel, iterations=3)
        image_2 = cv2.erode(img_bin, hor_kernel, iterations=3)
        horizontal_lines = cv2.dilate(image_2, hor_kernel, iterations=3)
        img_vh = cv2.addWeighted(vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
        img_vh = cv2.erode(~img_vh, kernel, iterations=2)
        thresh, img_vh = cv2.threshold(img_vh, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        bitxor = cv2.bitwise_xor(img, img_vh)
        self.bit_not = cv2.bitwise_not(bitxor)
        return img_vh

    def pre_process_image_good_structure(self) -> Optional[Tuple[ndarray, Tuple[ndarray]]]:
        """
        Выполнение различных операций с изображением и сегментация текста.
        :return: Изображение в виде матрицы.
        """
        img: ndarray = cv2.imread(self.file, 0)
        self.read_config_file(pytesseract.image_to_data(img, output_type='dict', lang='rus+eng'))
        img_vh = self.process_image_and_lines(img)
        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, boundingBoxes = self.sort_contours(contours, method="top-to-bottom")
        return img, contours

    @staticmethod
    def sort_contours(
            contours: Sequence[Union[cv2.Mat, ndarray]],
            method="left-to-right"
    ) -> Tuple[Tuple[ndarray], list]:
        """
        Сортируем контуры.
        :param contours: Контуры.
        :param method: Метод.
        :return: Контуры.
        """
        reverse: bool = method in ["right-to-left", "bottom-to-top"]
        i: int = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0
        boundingBoxes: list = [cv2.boundingRect(c) for c in contours]
        contours, boundingBoxes = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
        return contours, boundingBoxes

    def pre_process_image_bad_structure(self) -> Optional[Tuple[ndarray, list]]:
        """
        Выполнение различных операций с изображением и сегментация текста с нечеткой структурой таблицы.
        :return:
        """
        img: ndarray = cv2.imread(self.file, 0)
        reader: easyocr.Reader = easyocr.Reader(['ru', 'en'])
        contours: list = reader.readtext(img, paragraph=True, x_ths=8.0, y_ths=0.09)
        self.read_config_file(pytesseract.image_to_data(img, output_type='dict', lang='rus+eng'))
        self.bit_not = cv2.bitwise_not(img)
        return img, contours

    def get_all_contours_good(self, img: ndarray, contours: Tuple[ndarray]) -> Tuple[ndarray, list]:
        """

        :param img:
        :param contours:
        :return:
        """
        all_contours: list = []
        image: ndarray = img
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w != img.shape[1] and h != img.shape[0]) and h > self.min_height_of_cell and w > self.min_width_of_cell:
                all_contours.append([x, y, w, h])
                image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        return image, all_contours

    @staticmethod
    def get_all_contours_bad(img: ndarray, contours: Tuple[ndarray]) -> Tuple[ndarray, list]:
        """

        :param img:
        :param contours:
        :return:
        """
        all_contours: list = []
        image: ndarray = img
        for data in contours:
            box, text = data
            top_left, top_right, bottom_right, bottom_left = box
            list_top_left: list = [int(x) - 8 if i == 0 else int(x) for i, x in enumerate(top_left)]
            list_bottom_right: list = [int(x) + 8 if i == 0 else int(x) for i, x in enumerate(bottom_right)]
            list_width_height: list = [w - h for w, h in zip(bottom_right, top_left)]
            all_contours.append(list_top_left + list_width_height)
            image = cv2.rectangle(img, list_top_left, list_bottom_right, (0, 0, 0), 2)
        return image, all_contours

    def add_in_box_all_contours(self, image: ndarray, all_contours: Tuple[ndarray]) -> Tuple[list, list]:
        """
        Добавляем контуры в список.
        :param image: Изображение в виде матрицы.
        :param all_contours: Контуры.
        :return: Список контуров.
        """
        box: list = [Cell(image, *contour, self.bit_not, self.indent_x_text_of_cells, self.indent_y_text_of_cells,
                     self.config_for_pytesseract) for contour in all_contours]
        path: str = f"{os.path.dirname(self.output_directory_csv)}/img_structure"
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(f"{path}/{os.path.basename(self.file)}", image)
        return box, list(combinations(box, 2))

    @staticmethod
    def find_parent_and_child_in_table(list_contours_with_combinations: List[tuple]):
        """
        Находим родительские и дочерние ячейки в таблице.
        :param list_contours_with_combinations:
        :return:
        """
        for cell1, cell2 in list_contours_with_combinations:
            if cell1 > cell2:
                if cell2.parent is not None:
                    cell3: Cell = cell2.parent
                    if cell3 > cell1:
                        cell3.child.remove(cell2)
                        cell2.parent = cell1
                        cell1.child.append(cell2)
                else:
                    cell2.parent = cell1
                    cell1.child.append(cell2)

    @staticmethod
    def recognize_all_cells(box: list) -> list:
        """
        Распознаем текст в ячейках.
        :param box: Ячейка.
        :return: Ячейка.
        """
        for elem_of_box in box:
            elem_of_box.recognize_and_get_structure_of_table()
        return box

    def write_to_csv(self, box: list, contours, is_good_structure: bool) -> None:
        """
        Сохраняем данные в csv.
        :param box: Ячейка.
        :param contours:
        :param is_good_structure:
        :return:
        """
        if is_good_structure:
            for elem_of_box in box:
                df: Optional[pd.DataFrame] = elem_of_box.to_dataframe()
                if df is not None:
                    file: str = f"{self.file}_{elem_of_box.x1}_{elem_of_box.y1}.csv"
                    df.to_csv(f'{self.output_directory_csv}/{os.path.basename(file)}', encoding="utf-8", index=False)
        else:
            with open(f'{self.output_directory_csv}/{os.path.basename(self.file)}.csv', 'w', newline='', encoding='utf-8') \
                    as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Box_Top_Left', 'Box_Top_Right', 'Box_Bottom_Right', 'Box_Bottom_Left', 'Text'])
                for data in contours:
                    # box, text
                    box, text = data
                    top_left, top_right, bottom_right, bottom_left = box

                    tl = [int(x) for x in top_left]
                    tr = [int(x) for x in top_right]
                    br = [int(x) for x in bottom_right]
                    bl = [int(x) for x in bottom_left]

                    csv_writer.writerow([tl, tr, br, bl, text])

    def write_to_json(self, box: list) -> list:
        """
        Сохраняем в данные в json.
        :param box: Ячейка.
        :return: Данные.
        """
        list_all_table: list = [{"type": "label", "text": self.label_list}]
        for elem_of_box in box:
            if elem_of_box.parent:
                continue
            json_list: list = elem_of_box.to_json()
            list_all_table.append(json_list)
        with open(f'{self.output_directory}/{os.path.basename(f"{self.file}.json")}', "w", encoding="utf-8") as f:
            json.dump(list_all_table, f, ensure_ascii=False, indent=4, cls=CustomJSON)
        return list_all_table

    def main(self, is_good_structure: bool = False) -> None:
        """
        Основной метод, который запускает код.
        :return:
        """
        try:
            if "unknown.yml" in self.config_yaml_file.split("/")[-1]:
                return self.extracted_text()
            elif is_good_structure:
                img, contours = self.pre_process_image_good_structure()
                image, all_contours = self.get_all_contours_good(img, contours)
            else:
                img, contours = self.pre_process_image_bad_structure()
                image, all_contours = self.get_all_contours_bad(img, contours)
        except Exception as ex:
            logger.error(f"Exception is {ex}")
            return
        box, list_combinations = self.add_in_box_all_contours(image, all_contours)
        self.find_parent_and_child_in_table(list_combinations)
        self.recognize_all_cells(box)
        self.write_to_csv(box, contours, is_good_structure=is_good_structure)
        list_all_table: list = self.write_to_json(box)
        DataExtractor().parse_json(
            os.path.basename(self.file),
            os.path.dirname(self.output_directory_csv),
            list_all_table
        )


class Cell:
    """
    Класс для распознавания текста внутри каждой ячейки.
    """
    def __init__(self, img: ndarray, x1, y1, width, height, bit_not: ndarray, indent_x_text_of_cells: int,
                 indent_y_text_of_cells: int, config_for_pytesseract: str):
        self.img: ndarray = img
        self.parent: Optional[Cell] = None
        self.child: list = []
        self.x1: int = x1
        self.y1: int = y1
        self.x2: int = x1 + width
        self.y2: int = y1 + height
        self.bit_not: ndarray = bit_not
        self.indent_x_text_of_cells: int = indent_x_text_of_cells
        self.indent_y_text_of_cells: int = indent_y_text_of_cells
        self.config_for_pytesseract: str = config_for_pytesseract
        self.text: Optional[str] = None
        self.score: Optional[float] = None
        self.std: Optional[float] = None
        self.validations: Optional[bool] = None
        self.row: Optional[int] = None
        self.col: Optional[int] = None
        self.child_max_row: Optional[int] = None
        self.child_max_col: Optional[int] = None

    def __gt__(self, other):
        return all((self.x1 <= other.x1, self.y1 <= other.y1, self.x2 >= other.x2, self.y2 >= other.y2))

    def __str__(self):
        return str(self.x1, self.y1, self.x2, self.y2)

    def recognize(self, lang: str) -> str:
        """
        Распознаем текст внутри ячейки.
        :param lang: Язык для распознавания.
        :return: Распознанный текст.
        """
        if self.child:
            bit_not2 = self.bit_not.copy()
            for coordinates in self.child:
                cv2.rectangle(bit_not2, (coordinates.x1, coordinates.y1), (coordinates.x2, coordinates.y2), (255, 255,
                                                                                                             255), -1)
        else:
            bit_not2: ndarray = self.bit_not
        final_img: ndarray = bit_not2[self.y1 + self.indent_x_text_of_cells: self.y2 - self.indent_x_text_of_cells,
                                      self.x1 + self.indent_y_text_of_cells: self.x2 - self.indent_y_text_of_cells]

        kernel: ndarray = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        resizing: ndarray = cv2.resize(final_img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        dilation: ndarray = cv2.dilate(resizing, kernel, iterations=1)
        erosion: ndarray = cv2.erode(dilation, kernel, iterations=1)
        erosion = cv2.fastNlMeansDenoising(erosion, None, 20, 7, 21)
        text: dict = pytesseract.image_to_data(erosion, output_type="dict", lang=lang,
                                               config=self.config_for_pytesseract)
        dict_text, out, list_score = RecognizeTable.find_score_and_text(text, labels=False)
        inner: str = out.strip().translate({ord(c): " " for c in r"!@#$%^&*()[]{};<>?\|`~-=_+"})
        self.text = inner
        self.score = np.mean(list_score) if len(list_score) != 0 else None
        self.std = np.std(list_score) if len(list_score) != 0 else None
        self.validations = np.mean(list_score) > 85 if len(list_score) != 0 else None
        return inner

    def get_structure_of_row(self) -> None:
        """
        Получить структуру строк.
        :return:
        """
        if not self.child:
            return
        list_h: List[int] = [elem.y2 - elem.y1 for elem in self.child]
        min_h_of_cell: int = min(list_h)
        self.child.sort(key=lambda c: c.y1)
        current_row: int = 0
        prev_child: Cell = self.child[0]
        current_row_y1: int = prev_child.y1
        prev_child.row = current_row
        if len(self.child) > 1:
            for current_child in self.child[1:]:
                if current_child.y1 - current_row_y1 > min_h_of_cell * 0.62:
                    current_row += 1
                current_child.row = current_row
                current_row_y1 = current_child.y1
                prev_child = current_child
            prev_child.row = current_row
        self.child_max_row = current_row

    def get_structure_of_col(self) -> None:
        """
        Получить структуру колонок.
        :return:
        """
        if not self.child:
            return
        list_w: List[int] = [elem.x2 - elem.x1 for elem in self.child]
        min_w_of_cell: int = min(list_w)
        self.child.sort(key=lambda c: c.x1)
        current_col: int = 0
        prev_child: Cell = self.child[0]
        current_col_x1: int = prev_child.x1
        prev_child.col = current_col
        if len(self.child) > 1:
            for current_child in self.child[1:]:
                if current_child.x1 - current_col_x1 > min_w_of_cell * 0.62:
                    current_col += 1
                    current_col_x1 = current_child.x1
                current_child.col = current_col
                prev_child = current_child
            prev_child.col = current_col
        self.child_max_col = current_col

    def to_dataframe(self) -> Optional[pd.DataFrame]:
        """
        Сохраняем в таблицу типа DataFrame.
        :return:
        """
        if not self.child or (self.child_max_col == 1 and self.child_max_row == 1):
            return
        df: pd.DataFrame = pd.DataFrame(columns=range(self.child_max_col), index=range(self.child_max_row))
        for box in self.child:
            df.loc[box.row, box.col] = box.text
        return df

    def to_json(self) -> Union[list, dict]:
        """
        Сохраняем данные в json.
        :return: Данные в виде json.
        """
        predicted_boxes: dict = {
            "type": "text",
            "text": self.text, "row": self.row, "col": self.col, "xmin": self.x1, "ymin": self.y1,
            "xmax": self.x2, "ymax": self.y2, "score": self.score, "std": self.std, "is_valid": self.validations
        }
        json_list: list = [predicted_boxes]
        if not self.child:
            return predicted_boxes
        table: dict = {"type": "table", "cells": [child.to_json() for child in self.child]}
        json_list.append(table)
        return json_list

    def recognize_and_get_structure_of_table(self):
        """
        Распознаем структуру и текст ячеек.
        :return:
        """
        inner: str = self.recognize(lang="rus+eng")
        print(inner)
        if "Контейнеры" in inner:
            print(self.recognize(lang="eng"))
        self.get_structure_of_col()
        self.get_structure_of_row()


class DataExtractor:
    """
    Класс для извлечения нужных данных из полного перечня распознанного текста.
    """
    def __init__(self):
        self.date: Optional[str] = None
        self.ship: Optional[str] = None
        self.goods_name: Optional[str] = None
        self.tnved: Optional[int] = None
        self.consignee: Optional[str] = None
        self.shipper: Optional[str] = None
        self.containers: List[str] = []

    def validate_containers(self, dict_containers: dict) -> None:
        """
        Проверяем валидацию контейнеров.
        :param dict_containers: Словарь для сохранения контейнеров.
        :return:
        """
        for container in self.containers:
            container = re.sub(r"\s+", "", container.strip())
            if re.match(r"[A-Z]{4}\d{7}", container):
                match_container: Optional[Match[str]] = re.match(r"[A-Z]{4}\d{7}", container)
                dict_containers[match_container[0]].append(True)
            else:
                dict_containers[container].append(False)

    def get_data_from_cell(self, cell: Union[dict, list], dict_table: dict) -> None:
        """
        Получаем данные из ячеек.
        :param cell: Ячейка.
        :param dict_table: Словарь для сохранения данных из ячеек.
        :return:
        """
        if isinstance(cell, dict):
            if cell.get("type") == "table":
                self.get_data_from_cell(cell.get("cells"), dict_table)
            col_table = cell.get("col")
            row_table = cell.get("row")
            text = cell.get("text")
            dict_table[row_table, col_table] = text
            if col_table == 0 and bool(re.match(r'^(?=.*[a-zA-ZА-Яа-я])(?=.*\d).+$', text)):
                self.containers.append(text)
        elif isinstance(cell, list):
            for dict_cell in cell:
                self.get_data_from_cell(dict_cell, dict_table)

    def parse_json(self, file: str, directory: str, mobile_records: list) -> None:
        """
        Парсим полные данные из json.
        :param file: Наименование файла.
        :param directory: Наименование директории.
        :param mobile_records: Полные данные в json.
        :return:
        """
        dict_containers: defaultdict = defaultdict(list)
        index: int = 1
        for json_data in mobile_records:
            if isinstance(json_data, dict) and json_data["type"] == "label":
                for label in json_data["text"]:
                    self.date = label.get("text") if label.get("label") == "ShipAndVoyage" else None
            elif isinstance(json_data, list):
                for data in json_data:
                    dict_table: dict = {}
                    for cell in data.get("cells", []):
                        self.get_data_from_cell(cell, dict_table)
                    self.get_necessary_data(dict_table, index == 2)
                index += 1
        self.validate_containers(dict_containers)
        self.write_parsed_data_to_csv(file, directory, dict_containers)

    def get_necessary_data(self, dict_table: dict, is_normal_table: bool) -> None:
        """
        Получаем нужные данные, которые будут сохраняться в таблицу.
        :param dict_table: Словарь для извлечения данных.
        :param is_normal_table: Проверяем, нормальная ли структура таблицы.
        Есть таблица, у которой строки идут не снизу, а справа.
        :return:
        """
        if dict_table:
            self.consignee = self.get_key_by_value(self.consignee, dict_table, "Грузополучатель", is_normal_table)
            self.shipper = self.get_key_by_value(self.shipper, dict_table, "Грузоотправитель", is_normal_table)
            self.ship = self.get_key_by_value(self.ship, dict_table, "Наименование судна", is_normal_table)
            self.goods_name = self.get_key_by_value(self.goods_name, dict_table, "Наименование товара", is_normal_table)
            self.tnved = self.get_key_by_value(self.tnved, dict_table, "Код ТНВЭД товара", is_normal_table)

    @staticmethod
    def get_key_by_value(value_of_column: Optional[str], dict_table: dict, value_to_find: str, is_normal_table: bool) \
            -> Optional[str]:
        """
        Получаем нужную колонку и его значение по примерному совпадению (fuzz).
        :param value_of_column: Значение, которое сохраняется в таблицу.
        :param dict_table: Словарь для извлечения данных.
        :param value_to_find: Какую колонку должны найти в тексте.
        :param is_normal_table: Проверяем, нормальная ли структура таблицы.
        Есть таблица, у которой строки идут не снизу, а справа.
        :return:
        """
        for key, value in dict_table.items():
            if fuzz.ratio(value, value_to_find) > 80:
                key: tuple = (key[0] + 1, key[1]) if is_normal_table else (key[0], key[1] + 1)
                return dict_table[key]
        return value_of_column

    def write_parsed_data_to_csv(self, file: str, directory: str, dict_containers: dict) -> None:
        """
        Сохраняем отпарсенные данные в csv.
        :param file: Наименование файла.
        :param directory: Наименование директории.
        :param dict_containers: Данные с контейнерами.
        :return:
        """
        load_data: str = f"{directory}/csv_parsed"
        if not os.path.exists(load_data):
            os.makedirs(load_data)
        with open(f"{load_data}/{file}_parsed.csv", "w") as f:
            writer: csv.DictWriter = csv.DictWriter(f, fieldnames=['date', 'ship', 'goods_name', 'container_number',
                                                                   'tnved', 'is_valid', 'consignee', 'shipper'])
            writer.writeheader()
            for container, is_valid in dict_containers.items():
                for validation in is_valid:
                    writer.writerow(dict(date=self.date, ship=self.ship, goods_name=self.goods_name, tnved=self.tnved,
                                         container_number=container, is_valid=validation, consignee=self.consignee,
                                         shipper=self.shipper))


if __name__ == "__main__":
    ocr_image: HandleMultiPagesPDF = HandleMultiPagesPDF(sys.argv[1], sys.argv[2])
    ocr_image.main()
