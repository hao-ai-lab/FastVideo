## How to generate counting_edit dataset

1. Use [create\_prompts.py](https://github.com/djghosh13/geneval/blob/main/prompts/create_prompts.py) to create prompts for the counting task.
2. Run `python process_data.py` to create the edited data.

The edited data retains the Geneval format to facilitate using Geneval for evaluating counting as a reward. The meanings of the specific keys are as follows:

```
{
  "tag": "counting", 
  "include": [{"class": "clock", "count": 2}], 
  "exclude": [{"class": "clock", "count": 3}], 
  "t2i_prompt": "a photo of three clocks",  # Original prompt used to generate the image
  "prompt": "Change the number of clock in the image to two.",  # Editing prompt
  "image": "generated_images/image_43.jpg"  # The image generated using the t2i_prompt
}
```
