import re
import sys
import cv2
import csv
import json
import glob
import yaml
import math
import magic
import shutil
import requests
import psycopg2
import contextlib
import pytesseract
import numpy as np
import pandas as pd
import scipy.ndimage
from PIL import Image
from __init__ import *
from typing import Union
from PyPDF2 import PdfFileReader
from multiprocessing import Pool
from itertools import combinations
from collections import namedtuple
from collections import defaultdict
from pdf2image import convert_from_path
from configuration import Configuration
from validations_post_processing import ValidationsAndPostProcessing


logger: getLogger = get_logger("logging")


class HandleMultiPagesPDF:
    def __init__(self, file: str, dir_cache: str):
        self.file: str = file
        self.dir_cache: str = dir_cache

    def get_count_pages_from_file(self) -> int:
        """

        :param:
        :return:
        """
        with open(self.file, 'rb') as fl:
            page_num: int = PdfFileReader(fl).getNumPages()
            logger.info(f"Количество файлов pdf в многостраничном файле: {page_num}")
            return page_num

    def split_file(self):
        """

        :return:
        """
        images = convert_from_path(self.file)
        for i in range(len(images)):
            images[i].save(f'{self.dir_cache}/{os.path.basename(self.file)}_{str(i)}.jpg', 'JPEG')

    def get_count_split_pages(self):
        """

        :return:
        """
        path, dirs, files = next(os.walk(self.dir_cache))
        file_count: int = len(files)
        logger.info(f"Количество разбитых файлов pdf в кэше: {file_count}")

    def start_file_processing(self):
        """

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

    def handle_pages(self, file: str, dir_main: str, dir_csv: str, dir_json: str, classification: str):
        """

        :param file:
        :param dir_main:
        :param dir_csv:
        :param dir_json:
        :param classification:
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

    def main(self):
        """

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
            logger.error(f"The file format is not supported {os.path.basename(self.file)}")
        self.start_file_processing()


class HandleJPG:
    def __init__(self, input_file, dir_main, classification, configs):
        self.input_file = input_file
        self.dir_main = dir_main
        self.classification = classification
        self.configs = configs

    @staticmethod
    def rotate(image, angle, background):
        old_width, old_height = image.shape[:2]
        angle_radian = math.radians(angle)
        width = abs(np.sin(angle_radian) * old_height) + abs(np.cos(angle_radian) * old_width)
        height = abs(np.sin(angle_radian) * old_width) + abs(np.cos(angle_radian) * old_height)

        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_mat[1, 2] += (width - old_width) / 2
        rot_mat[0, 2] += (height - old_height) / 2
        return cv2.warpAffine(image, rot_mat, (int(round(height)), int(round(width))), borderValue=background)

    def turn_img(self):
        if not os.path.exists(self.classification):
            os.makedirs(os.path.join(self.classification, "line"))
            os.makedirs(os.path.join(self.classification, "port"))
            os.makedirs(os.path.join(self.classification, "contract"))
            os.makedirs(os.path.join(self.classification, "unknown"))

        im = cv2.imread(str(self.input_file))
        rotate_img = pytesseract.image_to_osd(im, config='--psm 0 -c min_characters_to_try=5')
        angle_rotated_image = int(re.search(r'(?<=Orientation in degrees: )\d+', rotate_img)[0])
        rotated = self.rotate(im, angle_rotated_image, (0, 0, 0))
        file_name = os.path.basename(self.input_file)
        cv2.imwrite(f'{os.path.dirname(self.input_file)}/{file_name}', rotated)
        logger.info(f'Rotate: {angle_rotated_image}, Filename: {os.path.basename(file_name)}')
        with contextlib.suppress(Exception):
            requests.post("http://127.0.0.1:5000/get_logs", json={
                "logs": "Поворот изображений на прямой угол",
                "value": angle_rotated_image
            })

    def correct_skew(self, delta, limit):
        def determine_score(arr, angle):
            data = scipy.ndimage.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return score // 100000000

        def golden_ration(l, r, delta):
            phi = (1 + math.sqrt(5)) / 2
            resphi = 2 - phi
            x1 = l + resphi * (r - l)
            x2 = r - resphi * (r - l)
            f1 = determine_score(thresh, x1)
            f2 = determine_score(thresh, x2)
            scores = []
            angles = []
            while abs(r - l) > delta:
                if f1 < f2:
                    l = x1
                    x1 = x2
                    f1 = f2
                    x2 = r - resphi * (r - l)
                    f2 = determine_score(thresh, x2)
                    scores.append(f2)
                    angles.append(x2)
                else:
                    r = x2
                    x2 = x1
                    f2 = f1
                    x1 = l + resphi * (r - l)
                    f1 = determine_score(thresh, x1)
                    scores.append(f1)
                    angles.append(x1)
            return (x1 + x2) / 2

        image = cv2.imread(self.input_file)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        X_Y = namedtuple("X_Y", "x y")
        dict_angle_and_score = {0: X_Y(0, determine_score(thresh, 0))}
        best_angle = 0
        for angle in range(1, limit, delta):
            dict_angle_and_score[angle] = X_Y(angle, determine_score(thresh, angle))
            dict_angle_and_score[-angle] = X_Y(-angle, determine_score(thresh, -angle))
            sorted_x_y = sorted(dict_angle_and_score.values(), key=lambda xy: xy.y)
            max_value = sorted_x_y[-1]
            min_value = sorted_x_y[0]
            if max_value.y > min_value.y * 10:
                left = dict_angle_and_score.get(max_value.x - 1)
                right = dict_angle_and_score.get(max_value.x + 1)
                if left and right:
                    best_angle = golden_ration(left.x, right.x, 0.1)
                    best_score = determine_score(thresh, best_angle)
                    if best_score > min_value.y * 100:
                        break
                    else:
                        del dict_angle_and_score[max_value.x]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        corrected = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        logger.info(f'Skew is: {best_angle:.04f}, Filename: {os.path.basename(self.input_file)}')
        cv2.imwrite(f"{os.path.dirname(self.input_file)}/{os.path.basename(self.input_file)}", corrected)
        with contextlib.suppress(Exception):
            requests.post("http://127.0.0.1:5000/get_logs", json={
                "logs": "Выравнивание изображений под маленьким углом",
                "value": corrected
            })

    def move_file_in_dir(self, str_of_doc, file_name, predict=None):
        with open(f"{self.configs}/classification.yml", "r") as stream:
            try:
                yaml_file = yaml.safe_load(stream)
                for label_in_config in yaml_file:
                    if yaml_file[label_in_config]["key"].upper() in str_of_doc.upper():
                        with contextlib.suppress(OSError):
                            os.remove(yaml_file[label_in_config]['folder'] + '/' + os.path.basename(file_name))
                        if os.path.isfile(file_name):
                            shutil.copy2(file_name, f'{os.path.dirname(self.dir_main)}' 
                                                    f'/{yaml_file[label_in_config]["folder"]}')
                            os.remove(file_name)
                        else:
                            shutil.move(file_name, f'{os.path.dirname(self.dir_main)}'
                                                   f'/{yaml_file[label_in_config]["folder"]}')
                        predict = yaml_file[label_in_config]['name']

                if not predict:
                    logger.info(f'Filename: {os.path.basename(self.input_file)}, text: {str_of_doc}')
                    with contextlib.suppress(OSError):
                        os.remove(yaml_file[label_in_config]['folder'] + '/' +
                                  os.path.basename(file_name))
                    if os.path.isfile(file_name):
                        shutil.copy2(file_name, f'{os.path.dirname(self.dir_main)}/'
                                                f'{yaml_file["unknown"]["folder"]}')
                        os.remove(file_name)
                    else:
                        shutil.move(file_name, f'{os.path.dirname(self.dir_main)}/'
                                               f'{yaml_file["unknown"]["folder"]}')
                    predict = yaml_file["unknown"]['name']
            except yaml.YAMLError as exc:
                print("Exception", exc)
        return predict

    def classification_img(self):
        image = Image.open(self.input_file)
        ocr_df = pytesseract.image_to_data(image, output_type='data.frame', lang='rus+eng')
        float_cols = ocr_df.select_dtypes('float').columns
        ocr_df[float_cols] = ocr_df[float_cols].round(0).astype(int)
        ocr_df = ocr_df.replace(r'^\s*$', np.nan, regex=True)
        ocr_df = ocr_df.dropna().reset_index(drop=True)
        words = list(ocr_df.text)
        str_of_doc = " ".join(words[:20])
        predict = self.move_file_in_dir(str_of_doc, self.input_file)
        logger.info(f'Filename: {os.path.basename(self.input_file)}, Predict class: {predict}')
        with contextlib.suppress(Exception):
            requests.post("http://127.0.0.1:5000/get_logs", json={
                "logs": "Классификация изображений",
                "value": predict
            })
        return predict

    def main(self):
        self.turn_img()
        self.correct_skew(delta=1, limit=60)
        return self.classification_img()


class CustomJSONizer(json.JSONEncoder):
    def default(self, obj):
        return super().encode(bool(obj)) \
            if isinstance(obj, np.bool_) \
            else super().default(obj)


class RecognizeTable:
    label_list = []
    bitnot = None

    length_of_kernel = None
    min_height_of_cell = None
    min_width_of_cell = None
    indent_x_text_of_cells = None
    indent_y_text_of_cells = None
    config_for_pytesseract = None

    database = None
    user = None
    password = None
    host = None
    port = None
    table = None

    def __init__(self, file, output_directory, output_directory_csv, config_yaml_file,
                 scripts_for_validations_and_postprocessing):
        self.file = file
        self.output_directory = output_directory
        self.output_directory_csv = output_directory_csv
        self.config_yaml_file = config_yaml_file
        self.scripts_for_validations_and_postprocessing = scripts_for_validations_and_postprocessing

    @staticmethod
    def remove_duplicate_empty_lines(lst):
        result_text = []
        result_top = []
        result_left = []
        result_width = []
        prev_empty = False
        for list_text, list_top, list_left, list_width in zip(lst["text"], lst["top"], lst["left"], lst["width"]):
            if list_text.strip() == '' and prev_empty:
                continue
            result_text.append(list_text)
            result_top.append(list_top)
            result_left.append(list_left)
            result_width.append(list_width)
            prev_empty = list_text.strip() == ''
        return result_text, result_top, result_left, result_width

    @staticmethod
    def add_values_in_dict(ocr_json_label, dict_text, value):
        ocr_json_label["text"] = value
        ocr_json_label["xmin"] = dict_text[value][0]
        ocr_json_label["ymin"] = dict_text[value][1]
        ocr_json_label["xmax"] = dict_text[value][2]
        ocr_json_label["ymax"] = dict_text[value][3]
        ocr_json_label["score"] = dict_text[value][4]
        ocr_json_label["std"] = dict_text[value][5]

    @staticmethod
    def split_text_paragraphs(data, threshold):
        str_data = ""
        for i in range(len(data["text"]) - 1):
            if data["text"][i]:
                str_data += f"{data['text'][i + 1]} "
            elif data["top"][i + 1] - data["top"][i - 1] > 15:
                prefix = "\n"
                if data["right"][i - 1] > threshold:
                    prefix = ""
                str_data += f"{prefix}{data['text'][i + 1]} "
            else:
                str_data += f"{data['text'][i + 1]} "
        return str_data

    def write_to_file(self, dir_txt, str_data):
        with open(f'{dir_txt}/{os.path.basename(f"{self.file}.txt")}', "w", encoding="utf-8") as f:
            f.write(str_data)
            logger.info(f'Write to {dir_txt}/{os.path.basename(f"{self.file}.text")}')
            return True

    def convert_image_to_text(self):
        img = cv2.imread(self.file, 1)
        dir_txt = f"{os.path.dirname(self.output_directory)}/txt"
        if not os.path.exists(dir_txt):
            os.makedirs(dir_txt)
        noiseless_image_colored = cv2.fastNlMeansDenoisingColored(img, None, 20, 20, 7, 21)
        data = pytesseract.image_to_string(noiseless_image_colored, output_type="dict", lang='rus+eng')
        return self.write_to_file(dir_txt, data["text"])

    @staticmethod
    def find_score_and_text(data, labels=None):
        boxes = len(data['level'])
        list_i = []
        dict_text = {}
        list_score = []
        text_ocr = ''
        for i in range(boxes):
            (x, y, w, h) = (data["left"][i], data["top"][i], data["width"][i], data["height"][i])
            if data["text"][i] and not list_i[-1]:
                text_ocr = str(data["text"][i]) if labels else " ".join(data["text"])
                x_min = x
                y_min = y
                list_score.clear()
                list_score.append(data["conf"][i])
            if data["text"][i] and list_i[-1] and labels:
                text_ocr += " " + str(data["text"][i])
                list_score.append(data["conf"][i])
                x_max = x + w
                y_max = y + h
            elif not data["text"][i]:
                with contextlib.suppress(Exception):
                    dict_text[text_ocr] = (x_min, y_min, x_max, y_max, np.mean(list_score), np.std(list_score))
            list_i.append(data["text"][i])

        return dict_text, text_ocr, list_score

    def load_config_file(self, data):
        with open(self.config_yaml_file, "r") as stream:
            try:
                yaml_file = yaml.safe_load(stream)
                self.length_of_kernel, self.min_height_of_cell, self.min_width_of_cell, self.indent_x_text_of_cells, \
                    self.indent_y_text_of_cells, self.config_for_pytesseract = \
                    Configuration().config_of_tables(yaml_file)
                self.database, self.user, self.password, self.host, self.port, self.table = \
                    Configuration().config_of_database(yaml_file)
                dict_text = self.find_score_and_text(data, labels=True)[0]
                for value in list(dict_text.keys()):
                    for len_label_in_config in range(len(yaml_file["labels"])):
                        ocr_json_label = {}
                        if yaml_file["labels"][len_label_in_config]["key"].upper() in value.upper():
                            ocr_json_label["label"] = yaml_file["labels"][len_label_in_config]["label"]
                            self.add_values_in_dict(ocr_json_label, dict_text, value)
                            postprocessing = ValidationsAndPostProcessing().postprocessing(yaml_file,
                                                                                           len_label_in_config,
                                                                                           self.scripts_for_validations_and_postprocessing,
                                                                                           value)
                            ocr_json_label["text"] = postprocessing
                            validations = ValidationsAndPostProcessing().validations(yaml_file, len_label_in_config,
                                                                                     self.scripts_for_validations_and_postprocessing,
                                                                                     postprocessing,
                                                                                     ocr_json_label["score"])
                            ocr_json_label["is_valid"] = bool(validations)
                            self.label_list.append(ocr_json_label)
            except yaml.YAMLError as exc:
                print(exc)
            except TypeError:
                return self.convert_image_to_text()

    def convert_image_in_black_white(self):
        img = cv2.imread(self.file, 0)
        with contextlib.suppress(Exception):
            if self.load_config_file(pytesseract.image_to_data(img, output_type='dict', lang='rus+eng')):
                return
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
        self.bitnot = cv2.bitwise_not(bitxor)
        contours, hierarchy = cv2.findContours(img_vh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours, boundingBoxes = self.sort_contours(contours, method="top-to-bottom")
        return img, contours, self.bitnot, self.indent_x_text_of_cells, self.indent_y_text_of_cells, self.config_for_pytesseract

    def sort_contours(self, cnts, method="left-to-right"):
        reverse = method in ["right-to-left", "bottom-to-top"]
        i = 1 if method in ["top-to-bottom", "bottom-to-top"] else 0
        # construct the list of bounding boxes and sort them from top to
        # bottom
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        cnts, boundingBoxes = zip(*sorted(zip(cnts, boundingBoxes),
                                          key=lambda b: b[1][i], reverse=reverse))
        # return the list of sorted contours and bounding boxes
        return cnts, boundingBoxes

    def add_in_box_all_contours(self, img, contours, bitnot, indent_x_text_of_cells, indent_y_text_of_cells, config_for_pytesseract):
        all_contours = []
        image = img
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if (w != img.shape[1] and h != img.shape[0]) and h > self.min_height_of_cell and w > self.min_width_of_cell:
                all_contours.append([x, y, w, h])
                image = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        box = [Cell(img, *contour, bitnot, indent_x_text_of_cells, indent_y_text_of_cells, config_for_pytesseract) for contour in
               all_contours]
        cv2.imwrite(self.output_directory_csv + "/" + os.path.basename(f"{self.file}"), image)
        return box, list(combinations(box, 2))

    def find_parent_and_childs_in_table(self, list_contours_with_combinations):
        for cell1, cell2 in list_contours_with_combinations:
            if all(cell1 > cell2):
                if cell2.parent is not None:
                    cell3 = cell2.parent
                    if all(cell3 > cell1):
                        cell3.childs.remove(cell2)
                        cell2.parent = cell1
                        cell1.childs.append(cell2)
                else:
                    cell2.parent = cell1
                    cell1.childs.append(cell2)

    def recognize_all_cells(self, box):
        for elem_of_box in box:
            elem_of_box.recognize_and_get_structure_of_table()
        return box

    def write_to_csv(self, box):
        for elem_of_box in box:
            df = elem_of_box.to_dataframe()
            if df is not None:
                df.to_csv(self.output_directory_csv + "/" + os.path.basename(f"{self.file}_{elem_of_box.x1}"
                                                                             f"_{elem_of_box.y1}.csv"),
                          encoding="utf-8", index=False)

    def write_to_json(self, box):
        list_all_table = [{"type": "label", "text": self.label_list}]
        for elem_of_box in box:
            if elem_of_box.parent:
                continue
            json_list = elem_of_box.to_json()
            list_all_table.append(json_list)

        with open(self.output_directory + "/" + os.path.basename(f"{self.file}.json"), "w", encoding="utf-8") as f:
            json.dump(list_all_table, f, ensure_ascii=False, indent=4, cls=CustomJSONizer)

        return list_all_table

    def push_to_db(self, list_all_table):
        try:
            conn = psycopg2.connect(database=self.database, user=self.user, password=self.password, host=self.host,
                                    port=self.port)
            cur = conn.cursor()
            data_json = json.dumps(list_all_table, ensure_ascii=False, indent=4, cls=CustomJSONizer)
            sql = f"INSERT INTO {self.table} (image, url_image, data_json) VALUES (%s, %s, %s)"
            val = (f'upload/{os.path.basename(self.file)}', f'{os.path.basename(self.file)}', data_json)
            cur.execute(sql, val)
            cur.close()
            conn.commit()
            conn.close()
        except Exception as exception:
            print(exception)

    def main(self):
        try:
            img, contours, bitnot, indent_x_text_of_cells, indent_y_text_of_cells, config_for_pytesseract = self.convert_image_in_black_white()
        except Exception as ex:
            print(ex)
            return
        box, list_combinations = self.add_in_box_all_contours(img, contours, bitnot, indent_x_text_of_cells, indent_y_text_of_cells, config_for_pytesseract)
        self.find_parent_and_childs_in_table(list_combinations)
        self.recognize_all_cells(box)
        self.write_to_csv(box)
        list_all_table = self.write_to_json(box)
        self.push_to_db(list_all_table)
        CSVData().parsed_json_from_db(os.path.basename(self.file), self.output_directory_csv, list_all_table)


class Cell:
    def __init__(self, img, x1, y1, width, height, bitnot, indent_x_text_of_cells, indent_y_text_of_cells, config_for_pytesseract):
        self.img = img
        self.parent = None
        self.childs = []
        self.x1 = x1
        self.y1 = y1
        self.x2 = x1 + width
        self.y2 = y1 + height
        self.bitnot = bitnot
        self.indent_x_text_of_cells = indent_x_text_of_cells
        self.indent_y_text_of_cells = indent_y_text_of_cells
        self.config_for_pytesseract = config_for_pytesseract
        self.text = None
        self.score = None
        self.std = None
        self.validations = None
        self.row = None
        self.col = None
        self.child_max_row = None
        self.child_max_col = None

    def __gt__(self, other):
        return self.x1 <= other.x1, self.y1 <= other.y1, self.x2 >= other.x2, self.y2 >= other.y2

    def __str__(self):
        coords = self.x1, self.y1, self.x2, self.y2
        return str(coords)

    def __repr__(self):
        return self.__str__()

    def recognize(self, lang) -> str:
        if self.childs:
            bitnot2 = self.bitnot.copy()
            for coordinates in self.childs:
                cv2.rectangle(bitnot2, (coordinates.x1, coordinates.y1), (coordinates.x2, coordinates.y2), (255, 255, 255), -1)
        else:
            bitnot2 = self.bitnot
        finalimg = bitnot2[self.y1 + self.indent_x_text_of_cells : self.y2 - self.indent_x_text_of_cells, (self.x1 + self.indent_y_text_of_cells) : self.x2 - self.indent_y_text_of_cells]

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
        resizing = cv2.resize(finalimg, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        dilation = cv2.dilate(resizing, kernel, iterations=1)
        erosion = cv2.erode(dilation, kernel, iterations=1)
        erosion = cv2.fastNlMeansDenoising(erosion, None, 20, 7, 21)
        text = pytesseract.image_to_data(erosion, output_type="dict", lang=lang, config=self.config_for_pytesseract)

        dict_text, out, list_score = RecognizeTable.find_score_and_text(text, labels=False)
        inner = out.strip()
        inner = inner.translate({ord(c): " " for c in "!@#$%^&*()[]{};<>?\|`~-=_+"})
        self.text = inner
        self.score = np.mean(list_score) if len(list_score) != 0 else None
        self.std = np.std(list_score) if len(list_score) != 0 else None
        self.validations = np.mean(list_score) > 85 if len(list_score) != 0 else None
        return inner

    def get_structure_of_row(self):
        if not self.childs:
            return
        list_h = [elem.y2 - elem.y1 for elem in self.childs]
        min_h_of_cell = min(list_h)
        self.childs.sort(key=lambda c: c.y1)
        current_row = 0
        prev_child = self.childs[0]
        current_row_y1 = prev_child.y1
        prev_child.row = current_row
        if len(self.childs) > 1:
            for current_child in self.childs[1:]:
                if current_child.y1 - current_row_y1 > min_h_of_cell * 0.62:
                    current_row += 1
                current_child.row = current_row
                current_row_y1 = current_child.y1
                prev_child = current_child
            prev_child.row = current_row
        self.child_max_row = current_row

    def get_structure_of_col(self):
        if not self.childs:
            return
        list_w = [elem.x2 - elem.x1 for elem in self.childs]
        min_w_of_cell = min(list_w)
        self.childs.sort(key=lambda c: c.x1)
        current_col = 0
        prev_child = self.childs[0]
        current_col_x1 = prev_child.x1
        prev_child.col = current_col
        if len(self.childs) > 1:
            for current_child in self.childs[1:]:
                if current_child.x1 - current_col_x1 > min_w_of_cell * 0.62:
                    current_col += 1
                    current_col_x1 = current_child.x1
                current_child.col = current_col
                prev_child = current_child
            prev_child.col = current_col
        self.child_max_col = current_col

    def to_dataframe(self) -> object:
        if not self.childs or (self.child_max_col == 1 and self.child_max_row == 1):
            return
        df = pd.DataFrame(columns=range(self.child_max_col), index=range(self.child_max_row))
        for box in self.childs:
            df.loc[box.row, box.col] = box.text
        return df

    def to_json(self) -> Union[list, dict]:
        predicted_boxes_dict = {
            "type": "text",
            "text": self.text, "row": self.row, "col": self.col, "xmin": self.x1, "ymin": self.y1,
            "xmax": self.x2, "ymax": self.y2, "score": self.score, "std": self.std, "is_valid": self.validations
        }
        json_list = [predicted_boxes_dict]
        if not self.childs:
            return predicted_boxes_dict
        table = {"type": "table", "cells": [child.to_json() for child in self.childs]}
        json_list.append(table)
        return json_list

    def recognize_and_get_structure_of_table(self):
        inner = self.recognize(lang="rus+eng")
        print(inner)
        if "Контейнеры" in inner:
            print(self.recognize(lang="eng"))
        self.get_structure_of_col()
        self.get_structure_of_row()


class CSVData:

    def __init__(self):
        self.consignee = None
        self.shipper = None
        self.containers = []

    def iter_all_containers(self, dict_containers):
        for container in self.containers:
            container = container.strip()
            try:
                if len(container.split()[0]) < 11: container = container.replace(" ", "")
            except IndexError:
                continue
            if re.match("[A-Z]{4}\d{7}", container):
                for container_number in container.split():
                    if re.match("[A-Z]{4}\d{7}", container_number):
                        con = re.match("[A-Z]{4}\d{7}", container_number)
                        dict_containers[con[0]].append(True)
                    elif len(container_number) > 10:
                        dict_containers[container_number].append(False)
            else:
                dict_containers[container].append(False)

    def parse_json(self, cell, dict_table):
        if isinstance(cell, dict):
            if cell.get("type") == "text":
                col_table = cell.get("col")
                row_table = cell.get("row")
                text = cell.get("text")
                dict_table[row_table, col_table] = text
                if col_table == 0 and bool(re.match(r'^(?=.*[a-zA-ZА-Яа-я])(?=.*\d).+$', text)):
                    self.containers.append(text)
            elif cell.get("type") == "table":
                self.parse_json(cell.get("cells"), dict_table)
        elif isinstance(cell, list):
            for dict_cell in cell:
                self.parse_json(dict_cell, dict_table)

    def parsed_json_from_db(self, file, directory, mobile_records):
        print("Print each row and it's columns values")
        dict_containers = defaultdict(list)
        for json_data in mobile_records:
            for data in json_data:
                with contextlib.suppress(Exception):
                    dict_table = {}
                    for cell in data["cells"]:
                        self.parse_json(cell, dict_table)
                    self.get_specific_data_from_json(dict_table)
        self.iter_all_containers(dict_containers)
        self.write_parsed_data_to_csv(file, directory, dict_containers)

    def get_specific_data_from_json(self, dict_table):
        self.consignee = self.get_key_by_value(self.consignee, dict_table, "Грузополучатель")
        self.shipper = self.get_key_by_value(self.shipper, dict_table, "Грузоотправитель")
        # elif re.findall("Ko[a-z]", text):
        #     container_name = re.findall("Ko[a-z]+", text)[0] + ":"
        #     containers = re.split(r'[,./]+', text.replace(container_name, ''))
        #     self.iter_all_containers(containers, dict_containers)

    @staticmethod
    def get_key_by_value(value_of_column, dict_table, value_to_find):
        for key, value in dict_table.items():
            if value == value_to_find:
                key = key[0], key[1] + 1
                return dict_table[key]
        return value_of_column

    def write_parsed_data_to_csv(self, file, directory, dict_containers):
        load_data_from_db = f"{directory}/csv"
        if not os.path.exists(load_data_from_db):
            os.makedirs(load_data_from_db)
        with open(f"{load_data_from_db}/{file}_parsed.csv", "w") as f:
            writer = csv.DictWriter(f, fieldnames=['container_number', 'is_valid', 'consignee', 'shipper'])
            writer.writeheader()
            for container, is_valid in dict_containers.items():
                for validation in is_valid:
                    writer.writerow(
                        dict(container_number=container, is_valid=validation, consignee=self.consignee,
                             shipper=self.shipper))


if __name__ == "__main__":
    ocr_image: HandleMultiPagesPDF = HandleMultiPagesPDF(sys.argv[1], sys.argv[2])
    ocr_image.main()
