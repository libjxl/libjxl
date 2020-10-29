% Copyright (c) the JPEG XL Project
%
% Licensed under the Apache License, Version 2.0 (the "License");
% you may not use this file except in compliance with the License.
% You may obtain a copy of the License at
%
%      http://www.apache.org/licenses/LICENSE-2.0
%
% Unless required by applicable law or agreed to in writing, software
% distributed under the License is distributed on an "AS IS" BASIS,
% WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
% See the License for the specific language governing permissions and
% limitations under the License.

pkg load image;

args = argv();

metric = args{1};
original_filename = args{2};
decoded_filename = args{3};

original = pfs_read_luminance(original_filename);
decoded = pfs_read_luminance(decoded_filename);

switch (metric)
  case "psnr"
    res = qm_pu2_psnr(original, decoded);
  case "ssim"
    res = qm_pu2_ssim(original, decoded);
  otherwise
    error(sprintf("unrecognized metric %s", metric));
end

printf("%f\n", res);
