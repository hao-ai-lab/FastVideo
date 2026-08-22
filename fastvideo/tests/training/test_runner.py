from unittest.mock import patch, MagicMock
from fastvideo.training.runner import main
import argparse

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
