# SPDX-License-Identifier: Apache-2.0
# Adapted from diffusers and HY-WorldPlay

# Copyright 2025 The Hunyuan Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from fastvideo.models.vaes.hunyuan15vae import AutoencoderKLHunyuanVideo15
from fastvideo.configs.models.vaes import Hunyuan15VAEConfig

class AutoencoderKLHYWorld(AutoencoderKLHunyuanVideo15):
    # TODO(mingjia): add temporal caching support for HYWorld VAE

    def __init__(
        self,
        config: Hunyuan15VAEConfig,
    ) -> None:
        AutoencoderKLHunyuanVideo15.__init__(self, config)

