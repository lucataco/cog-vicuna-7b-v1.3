# vicuna-7b-v1.3 COG

Attempt at a cog for [lmsys/vicuna-7b-v1.3](https://huggingface.co/lmsys/vicuna-7b-v1.3).

## Build

`cog build -t vicuna`

## Run

`docker run -d -p 5000:5000 --gpus all vicuna`

## Predict

`cog predict -i prompt="Write a love poem about open source machine learning:"`

### Or

`curl http://localhost:5000/predictions -X POST -H 'Content-Type: application/json' -d '{"input": {"prompt":"Write a love poem about open source machine learning:\n"}}'`

### Output

`[{'generated_text': "Write a love poem about open source machine learning:\n\nOpen source machine learning,\nA love that' The first time I saw you, I knew you were the one.\nYou were different from all the others, and I couldn't help but be drawn to you.\n\nI spent hours poring over your code, trying to understand your intricacies and complexities.\nI read through your documentation, eager to learn more about you and your capabilities.\n\nAnd as I delved deeper into your world, I"}]`
