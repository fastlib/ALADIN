#    Copyright 2021 HIP Applied Computer Vision Lab, Division of Medical Image Computing, German Cancer Research Center
#    (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import Tuple, Union, List
import numpy as np
from nnunetv2.imageio.base_reader_writer import BaseReaderWriter


class NumpyIO(BaseReaderWriter):
    """
    ONLY SUPPORTS 1D timeseries
    """

    supported_file_endings = [
        '.npy'
    ]

    def read_images(self, image_fnames: Union[List[str], Tuple[str, ...]], annotations=False) -> Tuple[np.ndarray, dict]:
        images = []
        for f in image_fnames:
            npy_img = np.load(f)
            assert npy_img.ndim == 1 or npy_img.ndim == 2 or (npy_img.ndim == 3 and annotations) or (npy_img.ndim == 4 and annotations), "Only 1D timeseries with one or more channels supported"

            if npy_img.ndim == 4 and annotations:
                images.append(npy_img)
            elif npy_img.ndim == 3 and annotations:
                images.append(npy_img[None, :].transpose((2, 0, 1, 3)))

            elif npy_img.ndim == 2:
                # channel to front, add additional dim so that we have shape (c, 1, 1, X)
                if annotations:
                    images.append(npy_img[None, None, :].transpose((2, 0, 1, 3)))
                else:
                    images.append(npy_img.transpose((1, 0))[:, None])
            elif npy_img.ndim == 1:
                # add 3 additional dims so that we have shape (1, 1, 1, X)
                images.append(npy_img[None, None, None])

        if not self._check_all_same([i.shape for i in images]):
            print('ERROR! Not all input images have the same shape!')
            print('Shapes:')
            print([i.shape for i in images])
            print('Image files:')
            print(image_fnames)
            raise RuntimeError()
        return np.vstack(images, dtype=np.float32, casting='unsafe'), {'spacing': (999, 999, 1)}

    def read_seg(self, seg_fname: str) -> Tuple[np.ndarray, dict]:
        return self.read_images((seg_fname, ), annotations=True)

    def write_seg(self, seg: np.ndarray, output_fname: str, properties: dict) -> None:
        np.save(output_fname, seg.astype(np.uint8, copy=False))