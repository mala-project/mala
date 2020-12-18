from optuna.trial import Trial


class OptunaParameter:
    """Represents an optuna parameter."""

    def __init__(self, opttype="float", name="", low=0, high=0, choices=[]):
        self.name = name
        self.high = high
        self.low = low
        self.opttype = opttype
        self.choices = choices

        # For now, only three types of hyperparameters are allowed:
        # Lists, floats and ints.
        if self.opttype != "float" and self.opttype != "int" and self.opttype != "categorical":
            raise Exception("Unsupported Hyperparameter type.")

    def get_parameter(self, trial: Trial):
        if self.opttype == "float":
            return self.get_float(trial)
        if self.opttype == "int":
            return self.get_int(trial)
        if self.opttype == "categorical":
            return self.get_categorical(trial)
        raise Exception("Wrong hyperparameter type.")

    def get_float(self, trial: Trial):
        if self.opttype == "float":
            return trial.suggest_float(self.name, self.low, self.high)
        else:
            raise Exception("Wrong hyperparameter type.")

    def get_int(self, trial: Trial):
        if self.opttype == "int":
            return trial.suggest_int(self.name, self.low, self.high)
        else:
            raise Exception("Wrong hyperparameter type.")

    def get_categorical(self, trial: Trial):
        if self.opttype == "categorical":
            return trial.suggest_categorical(self.name, self.choices)
        else:
            raise Exception("Wrong hyperparameter type.")
