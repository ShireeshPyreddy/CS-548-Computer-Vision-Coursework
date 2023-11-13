# MIT LICENSE
#
# Copyright 2023 Michael J. Reale
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and
# to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

###############################################################################
# IMPORTS
###############################################################################

from enum import Enum

class LBP_LABEL_TYPES(Enum):  
    UNIFORM = "Uniform"
    UNIFORM_ROT = "UniformRot"
    FULL = "Full"    

base_dir = "assign04"
image_dir = base_dir + "/" + "images"
ground_dir = base_dir + "/" + "ground"
out_dir = base_dir + "/" + "output"

inputImageFilenames = [
    "Image0.png",
    "Image1.png",
    "Image2.png",
    "Image3.png",
    "Image4.png",
    "Image5.png",
    "Image6.png"
]

