import inspect


class JSONSerializable:
    def __init__(self):
        pass

    def to_json(self):
        return self._standard_serializer()

    @classmethod
    def from_json(cls, json_dict):
        return cls._standard_deserializer(json_dict)

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

    @classmethod
    def _standard_deserializer(cls, json_dict):
        deserialized_object = cls()
        for key in json_dict:
            setattr(deserialized_object, key, json_dict[key])
        return deserialized_object
