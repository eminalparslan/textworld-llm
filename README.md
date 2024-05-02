# Fine-tuning Llama2 for the TextWorld environment

Fine-tuning LLMs with RL for the [TextWorld](https://www.microsoft.com/en-us/research/project/textworld/) environment.

Built on the starter code/framework from the "[First TextWorld Problems](https://competitions.codalab.org/competitions/21557)" 2019 challenge.

# Steps

Create virtual environment and install dependencies (use Python 3.10):

`$ python3 -m venv venv && source ./venv/bin/activate`

`$ pip3 install -r requirements.txt`

The directory `starting_k/` contains different agents:
- `sample_submission_random/`: Agent that chooses randomly from the list of admissable commands (it has this information at a significant penalty to its score). This came in the starting kit for the challenge.
- `sample_submission_lstm-dqn/`: Agent that uses an LSTM-DQN model to choose its actions. Needs to be trained with `train.py` first. This also came with the starting kit for the challenge.
- `submission_llm/`: Agent which uses an LLM to choose its actions. No training step, just using the common-sense knowledge of the model.
- `submission_rl_llm/`: Agent which uses a fine-tuned LLM to choose its actions. Trained with `train.py` first.
  - NOTE: training LLMs take a lot of time and compute. I trained Llama2-7B on 2 RTX 3090s for ~3-4 hours for desirable results.

This directory also contains `ingestion.py` which is used to test the agents as follows:

`$ python3 ingestion.py <submission directory> <directory of games to test on>`

This will output a `stats.json` file containing the scores the agent received for each game. You can get the overall score using `score.py`:

`$ python3 score.py stats.json <directory to store results>`

The `ingestion.py` and `score.py` scripts came from the starting kit, but were modified for ease-of-use.

If you want to play a TextWorld game yourself, simply run `play.py`.
