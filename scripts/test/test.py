import torch
from tests.pipelines.hunyuan_video.test_hunyuan_video import HunyuanVideoPipelineFastTests
from tests.pipelines.mochi.test_mochi import MochiPipelineFastTests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HunyuanTestCls =  HunyuanVideoPipelineFastTests()
HunyuanTestCls.test_inference()

MochiTestCls = MochiPipelineFastTests()
MochiTestCls.test_inference()