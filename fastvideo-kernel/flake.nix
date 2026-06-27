{
  description = "Flake for FastVideo Kernel Hub builds";

  inputs = {
    self.submodules = true;
    kernel-builder.url = "github:huggingface/kernels";
  };

  outputs =
    {
      self,
      kernel-builder,
    }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;
    };
}
