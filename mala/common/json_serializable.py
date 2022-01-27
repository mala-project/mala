from abc import ABC, abstractmethod
import inspect


class JSONSerializable(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def to_json(self):
        pass

    def _standard_serializer(self):
        data = {}
        members = inspect.getmembers(self,
                                     lambda a: not (inspect.isroutine(a)))
        for member in members:
            # Filter out all private members, builtins, etc.
            if member[0][0] != "_":
                data[member[0]] = member[1]
        json_dict = {"object": type(self).__name__,
                     "data": data}
        return json_dict
