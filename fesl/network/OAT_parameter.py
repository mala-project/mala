import oapackage as oa


class OA_Parameter:
    def __init__(self, opttype="categorical", name="", choices=[]):
        self.name = name
        self.opttype = opttype
        self.choices = choices
        self.num_choices=len(self.choices)

        if self.opttype != "categorical":
            raise Exception("Unsupported Hyperparameter type.")
    
    def get_parameter(self, trial, idx):
        if self.opttype == "categorical":
            return self.get_categorical(trial, idx)
        else:
            raise Exception("Wrong hyperparameter type.")

    def get_categorical(self, trial, idx):
            return self.choices[trial[idx]]
        

