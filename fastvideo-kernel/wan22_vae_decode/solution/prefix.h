#pragma once
#include <ATen/ATen.h>
#include <torch/csrc/distributed/c10d/ProcessGroup.hpp>
// Exchange chronological full-frame prefix slices for chronological row bands.
// Source rank r owns [floor(T*r/world), floor(T*(r+1)/world)).
at::Tensor prefix_temporal_to_rows(const at::Tensor& local,int64_t total_frames,
    int rank,int world,const c10::intrusive_ptr<c10d::ProcessGroup>& pg);
