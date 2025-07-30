import numpy as np

from aladin.selfreflection.base import ReflectionBase
from aladin.utils.morphological import closingcentered, openingcentered

import matplotlib.pyplot as plt

class NoiseReflection(ReflectionBase):
    def __init__(self, debug=False):
        print("Noise module initialized")
        self.debug = debug

        pass

    def plot(self, record):
        print("Plot noise in record", record.recordname)

    def correct(self, record):
        print("Correct noise in record", record.recordname)

        noise_logits = record.delineation["noise"]["logits"]
        noise_binary = np.zeros_like(noise_logits)
        noise_binary[noise_logits > 0.5] = 1

        noise_uncertainty = record.delineation["noise"]["uncertainty"]
        noise_uncertainty[noise_binary==1] = 0

        #noise_binary[noise_binary==1] = 1
        noise_binary[np.log(noise_uncertainty+0.0001) > -1] = 1

        #noise_binary = openingcentered(noise_binary, np.ones(int(0.5*record.fs)))
        noise_binary = closingcentered(noise_binary, np.ones(int(2*record.fs)))
        noise_binary = openingcentered(noise_binary, np.ones(int(2*record.fs)))

        record.ecg[noise_binary==1] = 0
        record.filtered_ecg[noise_binary==1] = 0
        record.filtered_ecg -= np.mean(record.filtered_ecg)
        record.normalized_ecg -= np.mean(record.normalized_ecg)

        record.noise_mask = noise_binary

        return record