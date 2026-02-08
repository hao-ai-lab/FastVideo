from unittest.mock import MagicMock
import fastvideo.distributed.parallel_state as ps

ps.get_sp_world_size = MagicMock(return_value=1)

from fastvideo.configs.pipelines import LingBotWorldI2V480PConfig
from fastvideo.models.dits.lingbotworld.model import LingBotWorldTransformer3DModel


def inspect_without_weights():
    config = LingBotWorldI2V480PConfig()
    model = LingBotWorldTransformer3DModel(config=config.dit_config, hf_config={})
    print("\n" + "=" * 80)
    for name, param in model.named_parameters():
        print(f"{name:<55} | {str(list(param.shape)):<20}")

    print("-" * 80)


if __name__ == "__main__":
    inspect_without_weights()
