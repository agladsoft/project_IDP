import re
import datetime


class ShipAndVoyage:

    @staticmethod
    def replace_symbols_and_letters(value: str) -> str:
        month_list = ["янв.", "февр.", "марта", "апр.", "мая", "июн.", "июл.", "авг.", "сент.", "окт.", "нояб.", "дек."]
        reg_exp = "\d{1,2}[^\S\n\t]+\w+.[^\S\n\t]+\d{4}"
        date = re.findall(reg_exp, value)[0].split()
        if date[1] in month_list:
            month_digit = month_list.index(date[1]) + 1
        date = datetime.datetime.strptime(f'{date[2]}-{str(month_digit)}-{date[0]}', "%Y-%m-%d")
        return str(date.date())

    @staticmethod
    def is_validate_for_ship_and_voyage(value: str) -> bool:
        return bool(re.findall("\d{4}-\d{1,2}-\d{1,2}", value))


class ShipAndVoyage2:

    @staticmethod
    def replace_word(value: str) -> str:
        return value.replace("на", "по")

    @staticmethod
    def is_validate_for_ship_and_voyage2(value: str) -> bool:
        return len(value.split()) == 3
