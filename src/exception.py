import sys


class CustomException(Exception):
    """
    This class will raise the custom exception with the detailed error message.
    It inherits the Exception class.
    input:
        error_message: error message
        error_detail: sys module
    methods:
        error_message_detail: This method will return the error message with the file name and line number.
    """

    def __init__(self, error_message, error_detail: sys):
        # calling the __init__ method of the parent class Exception
        super().__init__(error_message)
        # getting the detailed error message with file name and line number
        self.detailed_error_message = self.error_message_detail(
            error_message, error_detail)

    # overriding the __str__ method in exception class with the custom error message
    def __str__(self):
        return self.detailed_error_message

    def error_message_detail(self, error_message, error_detail: sys):
        """
        This function will return the error message with the file name and line number.
        input: 
            error: error message
            error_detail: sys module
        output:
            error_message: error message with file name and line number
        """
        # getting info about the error
        _, _, error_info = error_detail.exc_info()
        # getting the file name and line number
        file_name = error_info.tb_frame.f_code.co_filename
        # getting the line number
        line_number = error_info.tb_lineno
        # rewriting the error message with file name and line number
        detailed_error_message = """
        Script Name: {0}
        Line Number: {1}
        Error Message: {2}""".format(
            file_name, line_number, str(error_message))

        return detailed_error_message
