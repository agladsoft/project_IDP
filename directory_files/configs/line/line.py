import re
from typing import List, Any


class inn:

    @staticmethod
    def get_inn(value: str) -> str:
        try:
            return re.findall("\d{10}", re.findall("КОД ИНН \d{10}", value)[0])[0]
        except Exception:
            return value

    @staticmethod
    def is_valid_inn(value: str) -> bool:
        try:
            return len(value) == 10
        except Exception:
            return False


class tnved:

    @staticmethod
    def get_tnved(value: str) -> str:
        try:
            return re.findall("\d{10}", re.findall("КОД ТНВЭД \d{10}", value)[0])[0]
        except Exception:
            return value

    @staticmethod
    def is_valid_tnved(value: str) -> bool:
        try:
            return len(value) == 10
        except Exception:
            return False


if __name__ == "__main__":
    var_inn = inn.get_inn('КОД ИНН 3448003962 КОД ТНВЭД 2815110000')
    print(var_inn)
    print(inn.is_valid_inn(var_inn))

    var_tnved = tnved.get_tnved('КОД ИНН 3448003962 КОД ТНВЭД 2815110000')
    print(var_tnved)
    print(tnved.is_valid_tnved(var_tnved))