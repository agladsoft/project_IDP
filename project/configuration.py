import contextlib


class Configuration:

    @staticmethod
    def config_of_tables(yaml_file):
        with contextlib.suppress(KeyError):
            length_of_kernel = yaml_file['config_of_table']['length_of_kernel']
            min_height_of_cell = yaml_file['config_of_table']['min_height_of_cell']
            min_width_of_cell = yaml_file['config_of_table']['min_width_of_cell']
            indent_x_text_of_cells = yaml_file['config_of_table']['indent_x_text_of_cells']
            indent_y_text_of_cells = yaml_file['config_of_table']['indent_y_text_of_cells']
            config_for_pytesseract = yaml_file['config_of_table']['config_for_pytesseract']
            return length_of_kernel, min_height_of_cell, min_width_of_cell, indent_x_text_of_cells, \
                   indent_y_text_of_cells, config_for_pytesseract

    @staticmethod
    def config_of_database(yaml_file):
        with contextlib.suppress(KeyError):
            database = yaml_file['config_of_database']['database']
            user = yaml_file['config_of_database']['user']
            password = yaml_file['config_of_database']['password']
            host = yaml_file['config_of_database']['host']
            port = yaml_file['config_of_database']['port']
            table = yaml_file['config_of_database']['table']
            return database, user, password, host, port, table