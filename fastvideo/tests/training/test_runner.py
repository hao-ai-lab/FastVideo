from unittest.mock import patch, MagicMock
from fastvideo.training.runner import main
import argparse
from fastvideo.utils import build_parser

def test_runner_invokes_correct_class():
    with patch('fastvideo.training.runner.importlib.import_module') as mock_import_module:
        mock_module = MagicMock()
        mock_import_module.return_value = mock_module
        
        mock_pipeline_class = MagicMock()
        setattr(mock_module, "WanTrainingPipeline", mock_pipeline_class)
        
        mock_pipeline_instance = MagicMock()
        mock_pipeline_class.from_pretrained.return_value = mock_pipeline_instance
        
        args = argparse.Namespace(
            pipeline_module="fastvideo.training.wan_training_pipeline",
            pipeline_class="WanTrainingPipeline",
            pretrained_model_name_or_path="test_model"
        )
        
        main(args)
        
        mock_import_module.assert_called_once_with("fastvideo.training.wan_training_pipeline")
        mock_pipeline_class.from_pretrained.assert_called_once_with("test_model", args=args)
        mock_pipeline_instance.train.assert_called_once()


def test_runner_cli_argument_parsing():

    base_required_args = [
        "--data_path", "test_path",
        "--dataloader_num_workers", "0",
        "--num_height", "448",
        "--num_width", "832",
        "--num_frames", "61",
        "--train_batch_size", "1",
        "--num_latent_t", "16",
        "--output_dir", "outputs/test",
        "--learning_rate", "2e-6",
    ]

    # Test with underscores
    parser1 = build_parser()
    args1 = parser1.parse_args([
        "--pipeline_class", "WanDistillationPipeline",
        "--pipeline_module", "fastvideo.training.wan_distillation_pipeline",
        "--pretrained_model_name_or_path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        *base_required_args,
    ])
    assert args1.pipeline_class == "WanDistillationPipeline"
    assert args1.pipeline_module == "fastvideo.training.wan_distillation_pipeline"

    # Test with dashes
    parser2 = build_parser()
    args2 = parser2.parse_args([
        "--pipeline-class", "WanDistillationPipeline",
        "--pipeline-module", "fastvideo.training.wan_distillation_pipeline",
        "--pretrained-model-name-or-path", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        *base_required_args,
    ])
    assert args2.pipeline_class == "WanDistillationPipeline"
    assert args2.pipeline_module == "fastvideo.training.wan_distillation_pipeline"

