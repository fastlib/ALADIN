import numpy as np
import time
import os
import aladin._main as cpp_backend

def preprocess_batch(records: list):
    """
    Preprocess a batch of records in parallel.
    """
    #print("Preprocessing batch of records...")
    
    # Use OpenMP for parallel processing
    cpp_backend.preprocess_records(records)
    
    #print("Batch preprocessing completed.")

class MultiRecord():
    def __init__(self, records: list):
        self.records = records
        self.ecg = []
        for record in records:
            self.ecg.append(record.ecg)
        self.fs = records[0].fs
        self.recordname = "MULTIRECORD"

    def __getitem__(self, index):
        return self.records[index]

    def __len__(self):
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

class Record():
    def __init__(self, ecg: np.ndarray, fs: int, db: str, recordname: str, delineation = None):

        self.cpp_record = cpp_backend.Record(ecg, fs)
        self._db = db
        self._recordname = recordname
        self._diagnoses = []
        self._groundtruth = []

    def add_diagnosis(self, name, explanation, start, end):
        d = cpp_backend.Diagnosis(name, explanation, start, end)
        self.cpp_record.add_diagnosis(d)
    
    def add_subdiagnosis(self, name, explanation, start, end):
        d = cpp_backend.Diagnosis(name, explanation, start, end)
        self.cpp_record.add_subdiagnosis(d)

    @property
    def ecg(self):
        return self.cpp_record.ecg

    @property
    def normalized_ecg(self):
        return self.cpp_record.normalized_ecg

    @property
    def filtered_ecg(self):
        return self.cpp_record.filtered_ecg

    @property
    def ecg_no_qrst(self):
        return self.cpp_record.ecg_no_qrst
    
    @property
    def ecg_noise(self):
        return self.cpp_record.ecg_noise
    
    @property
    def ecg_bandpass(self):
        return self.cpp_record.ecg_bandpass

    @property
    def delineations(self):
        return self.cpp_record.delineations

    @property
    def fs(self):
        return self.cpp_record.fs

    @property
    def recordname(self):
        return self._recordname

    @property
    def diagnosis(self):
        return self.cpp_record.diagnosis

    @property
    def subdiagnosis(self):
        return self.cpp_record.subdiagnosis
    
    @property
    def qrs(self):
        return self.cpp_record.qrs
    
    @property
    def p(self):
        return self.cpp_record.p
    
    @property
    def t(self):
        return self.cpp_record.t

    @property
    def qrs_clusters(self):
        return self.cpp_record.qrs_clusters

    @property
    def p_clusters(self):
        return self.cpp_record.p_clusters

class RecordCollection():
    def __init__(self, records: list):
        self.cpp_collection = cpp_backend.RecordCollection()
        for record in records:
            self.cpp_collection.add_record(record.cpp_record)

    def __getitem__(self, index):
        return self.cpp_collection.get_record(index)

    def __len__(self):
        return self.cpp_collection.size()

    def preprocess(self):
        """
        Preprocess all records in the collection.
        """
        #print("Preprocessing record collection...")
        self.cpp_collection.preprocess()
        #print("Record collection preprocessing completed.")
    