from importlib import machinery


class ValidationsAndPostProcessing:

    @staticmethod
    def postprocessing(yaml_file, len_label_in_config, scripts_for_validations_and_postprocessing, value):
        try:
            class_name = yaml_file["labels"][len_label_in_config]["label"]
            method_name = yaml_file["labels"][len_label_in_config]["postprocessing"]
            imported = machinery.SourceFileLoader(method_name, scripts_for_validations_and_postprocessing).load_module()
            class_name = getattr(imported, class_name)
            postprocessing = getattr(class_name, method_name)
            return postprocessing(value)
        except KeyError as ex_key:
            print("Not found the key by name postprocessing", ex_key)
            return value

    @staticmethod
    def validations(yaml_file, len_label_in_config, scripts_for_validations_and_postprocessing, value, score):
        try:
            class_name = yaml_file["labels"][len_label_in_config]["label"]
            method_name = yaml_file["labels"][len_label_in_config]["validations"]
            imported = machinery.SourceFileLoader(method_name, scripts_for_validations_and_postprocessing).load_module()
            class_name = getattr(imported, class_name)
            validations = getattr(class_name, method_name)
            return validations(value)
        except KeyError as ex_key:
            print("Not found the key by name validations", ex_key)
            return score > 85