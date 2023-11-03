import contextlib
from datetime import datetime
import re

date_formats: tuple = ("%Y-%m-%d", "%d.%m.%Y")
month_list = ["янв", "февр", "марта", "апр", "мая", "июн", "июл", "авг", "сент", "окт", "нояб", "дек"]


def convert_date(original_string):
    for month in month_list:
        if month in original_string:
            month_index = month_list.index(month) + 1
            break
    else:
        print("Месяц не найден в строке.")
        month_index = None
    if month_index is not None:
        year = re.search(r'\d{4}', original_string).group()
        day = re.search(r'\d+', original_string).group()
        return f'{year}-{month_index:02d}-{int(day):02d}'


def convert_format_date(date: str):
    """
    Convert to a date type.
    """
    for date_format in date_formats:
        with contextlib.suppress(ValueError):
            return str(datetime.strptime(date, date_format).date())
    return None


class ShipAndVoyage:

    @staticmethod
    def replace_symbols_and_letters(value: str, label: str) -> str:
        pattern = f'от(.*?){label}'
        # value = "от 04.08.2023 Таможенный пост"
        match = re.search(pattern, value)[1].strip()
        if date := convert_format_date(match):
            return date
        date = re.sub(r'[ \W_]+', '', match)
        return convert_date(date)

    @staticmethod
    def is_validate_for_ship_and_voyage(value: str) -> bool:
        return bool(re.findall(r"\d{4}-\d{1,2}-\d{1,2}", value))


class ShipAndVoyage2:

    @staticmethod
    def replace_word(value: str) -> str:
        return value.replace("на", "по")

    @staticmethod
    def is_validate_for_ship_and_voyage2(value: str) -> bool:
        return len(value.split()) == 3
