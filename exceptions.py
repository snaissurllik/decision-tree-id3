class ModelError(Exception):
    pass


class ModelNotTrainedError(ModelError):
    pass


class InvalidDatasetError(ModelError):
    pass
