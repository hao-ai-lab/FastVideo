# SPDX-License-Identifier: Apache-2.0
"""Request model for updating settings."""

from pydantic import BaseModel


class SettingsUpdate(BaseModel):
    defaultModelId: str | None = None
    defaultModelIdT2v: str | None = None
    defaultModelIdI2v: str | None = None
    defaultModelIdT2i: str | None = None
    numInferenceSteps: int | None = None
    numFrames: int | None = None
    height: int | None = None
    width: int | None = None
    guidanceScale: float | None = None
    guidanceRescale: float | None = None
    fps: int | None = None
    seed: int | None = None
    numGpus: int | None = None
    ditCpuOffload: bool | None = None
    textEncoderCpuOffload: bool | None = None
    vaeCpuOffload: bool | None = None
    imageEncoderCpuOffload: bool | None = None
    useFsdpInference: bool | None = None
    enableTorchCompile: bool | None = None
    vsaSparsity: float | None = None
    tpSize: int | None = None
    spSize: int | None = None
    autoStartJob: bool | None = None
    datasetUploadPath: str | None = None
