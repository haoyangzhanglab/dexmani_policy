# Source acknowledgements

KineDex per-finger CNN layer recipe:
DinoMini00/KineDex_code, `diffusion_policy/model/vision/multi_image_obs_encoder.py`.
The five-finger whole-hand fusion is not copied; the local adapter keeps five tokens.

ECA-Net mechanism and reference comparison:
BangguWu/ECANet, `models/eca_module.py`. The new module implements ECA after
pooling, algebraically equivalent to the source operation followed by GAP.
The original ECA source license is reproduced below.

## ECA-Net — MIT License

Copyright (c) 2019 BangguWu, Qilong Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
