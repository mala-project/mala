"""Base class for all objects that need to be JSON serializable."""

import inspect


class JSONSerializable:
    """
    Base class for all objects that need to be JSON serializable.

    Implements "to_json" and "from_json" for serialization and deserialization,
    respectively. Other classes that also need to be JSON serializable have
    to inherit from this class and reimplement these methods, if necessary.
    """

    def __init__(self):
        pass

    def to_json(self):
        """
        Convert this object to a dictionary that can be saved in a JSON file.

        Returns
        -------
        json_dict : dict
            The object as dictionary for export to JSON.

        """
        return self._standard_serializer()

    @classmethod
    def from_json(cls, json_dict):
        """
        Read this object from a dictionary saved in a JSON file.

        Parameters
        ----------
        json_dict : dict
            A dictionary containing all attributes, properties, etc. as saved
            in the json file.

        Returns
        -------
        deserialized_object : JSONSerializable
            The object as read from the JSON file.

        """
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
