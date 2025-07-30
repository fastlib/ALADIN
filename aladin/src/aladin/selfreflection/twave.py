import numpy as np

from aladin.selfreflection.base import ReflectionBase

class TWaveReflection(ReflectionBase):
    def __init__(self, debug=False):
        print("TWaveReflection module initialized")
        pass

    def reflect(self, record):
        print("Reflecting on T-waves in record", record.recordname)
        return record
