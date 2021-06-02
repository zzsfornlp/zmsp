#

# base protocol for data io

class BaseDataItem:
    # =====
    # from name to cls

    _registered_mappings = {}  # name to class mapping

    @staticmethod
    def register(cls):
        name = cls.__name__
        mapping = BaseDataItem._registered_mappings
        assert name not in mapping, f"Repeated entry name: {cls}({name})"
        mapping[name] = cls
        return cls

    # =====
    # IO: wrapped versions are useful for optional fields/components

    def to_builtin(self, *args, **kwargs):
        raise NotImplementedError()

    @classmethod
    def from_builtin(cls, d):
        raise NotImplementedError()

    def to_builtin_wrapped(self, *args, **kwargs):
        class_name = self.__class__.__name__
        assert class_name in BaseDataItem._registered_mappings, f"Entry not found: {self.__class__}({class_name})"
        return {"_type": class_name, "_v": self.to_builtin(*args, **kwargs)}

    @staticmethod
    def from_builtin_wrapped(d):
        class_name, val = d["_type"], d["_v"]
        class_type = BaseDataItem._registered_mappings[class_name]
        return class_type.from_builtin(val)

# short-cut
data_type_reg = BaseDataItem.register
