
class OsuError(Exception):
    def __init__(self, message, error_type, file_name):
        super(OsuError, self).__init__(message)

        self.error_type = error_type
        self.file_name = file_name
