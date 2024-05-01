from typing import List, Dict, Any, Optional

from textworld import EnvInfos

from peft.tuners.lora import LoraConfig
from peft.auto import AutoPeftModelForCausalLM
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModelForCausalLM
import torch
import wandb
import os

from collections import deque


class CustomAgent:
    """ Template agent for the TextWorld competition. """

    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False

        self.system_prompt = "You are playing a text-based game with a cooking theme. You will receive observations about the current state of the game and respond with commands. Here are some example commands: 'examine counter', 'inventory', 'go north', 'pick up the knife', 'fry the apple on the stove', 'open door', 'look', and 'goal' to remind yourself of the goal. These commands might not work and there are many commands not listed here. When responding, first reason about the game state to decide the best action to reach the goal and then say 'command: <your command>'. Only respond with the command and don't say anything else, even when you are told your commands aren't recognized."

        # store the last 10 chats (+1 for system prompt)
        self.chat = deque(maxlen=11)

        self.current_episode = 0
        # how often model should be saved during training
        self.save_frequency = 50

        # store last query and response
        self.query = None
        self.response = None

    def train(self) -> None:
        """ Tell the agent it is in training mode. """
        self.mode = "train"

    def eval(self) -> None:
        """ Tell the agent it is in evaluation mode. """
        self.mode = "eval"

    def select_additional_infos(self) -> EnvInfos:
        """
        Returns what additional information should be made available at each game step.

        Requested information will be included within the `infos` dictionary
        passed to `CustomAgent.act()`. To request specific information, create a
        :py:class:`textworld.EnvInfos <textworld.envs.wrappers.filter.EnvInfos>`
        and set the appropriate attributes to `True`. The possible choices are:

        * `description`: text description of the current room, i.e. output of the `look` command;
        * `inventory`: text listing of the player's inventory, i.e. output of the `inventory` command;
        * `max_score`: maximum reachable score of the game;
        * `objective`: objective of the game described in text;
        * `entities`: names of all entities in the game;
        * `verbs`: verbs understood by the the game;
        * `command_templates`: templates for commands understood by the the game;
        * `admissible_commands`: all commands relevant to the current state;

        In addition to the standard information, game specific information
        can be requested by appending corresponding strings to the `extras`
        attribute. For this competition, the possible extras are:

        * `'recipe'`: description of the cookbook;
        * `'walkthrough'`: one possible solution to the game (not guaranteed to be optimal);

        Example:
            Here is an example of how to request information and retrieve it.

            >>> from textworld import EnvInfos
            >>> request_infos = EnvInfos(description=True, inventory=True, extras=["recipe"])
            ...
            >>> env = gym.make(env_id)
            >>> ob, infos = env.reset()
            >>> print(infos["description"])
            >>> print(infos["inventory"])
            >>> print(infos["extra.recipe"])

        Notes:
            The following information *won't* be available at test time:

            * 'walkthrough'

            Requesting additional infos comes with some penalty (called handicap).
            The exact penalty values will be defined in function of the average
            scores achieved by agents using the same handicap.

            Handicap is defined as follows
                max_score, has_won, has_lost,               # Handicap 0
                description, inventory, verbs, objective,   # Handicap 1
                command_templates,                          # Handicap 2
                entities,                                   # Handicap 3
                extras=["recipe"],                          # Handicap 4
                admissible_commands,                        # Handicap 5
        """
        request_infos = EnvInfos()
        request_infos.description = True
        request_infos.inventory = True
        request_infos.entities = True
        request_infos.verbs = True
        request_infos.extras = ["recipe"]
        return request_infos

    def _init(self) -> None:
        """ Initialize the agent. """
        self._initialized = True

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        model_id = "meta-llama/Llama-2-7b-chat-hf"

        if self.mode == "train":
            lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
            self.model = AutoModelForCausalLMWithValueHead.from_pretrained(
                model_id,
                peft_config=lora_config,
                quantization_config=bnb_config,
                device_map="auto",
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            # self.reference_model = create_reference_model(self.model)
            self.ppo_config = PPOConfig(
                batch_size=1,
                mini_batch_size=1,
                optimize_cuda_cache=True,
                kl_penalty="abs",
                log_with="wandb",
            )
            self.ppo_trainer = PPOTrainer(self.ppo_config, self.model, None, self.tokenizer)
        else:
            tuend_model_id = None
            tuned_model_id = "saved_models/ft-400"

            if tuned_model_id is not None:
                model_id = tuned_model_id

                self.model = AutoPeftModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                    # offload_folder="./offload"
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    device_map="auto",
                    quantization_config=bnb_config,
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def _start_episode(self, obs: List[str], infos: Dict[str, List[Any]]) -> None:
        """
        Prepare the agent for the upcoming episode.

        Arguments:
            obs: Initial feedback for each game.
            infos: Additional information for each game.
        """
        if not self._initialized:
            self._init()

        self._epsiode_has_started = True

    def _end_episode(self, obs: List[str], scores: List[int], infos: Dict[str, List[Any]]) -> None:
        """
        Tell the agent the episode has terminated.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game.
            infos: Additional information for each game.
        """
        self._epsiode_has_started = False

        self.chat.clear()

        if self.mode == "train" and self.current_episode % self.save_frequency == 0:
            checkpoint_dir = "./saved_models"
            finetune_base_name = "ft"
            if not os.path.isdir(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            checkpoint_path = f"{checkpoint_dir}/{finetune_base_name}-{self.current_episode}"
            print("****** Saving pretrained model *************************")
            self.ppo_trainer.save_pretrained(checkpoint_path, save_embedding_layer=True)

        self.current_episode += 1

        print("Episode is over:")
        print(f"\t{scores=}")

    def act(self, observations: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> Optional[List[str]]:
        """
        Acts upon the current list of observations.

        One text command must be returned for each observation.

        Arguments:
            obs: Previous command's feedback for each game.
            scores: The score obtained so far for each game.
            dones: Whether a game is finished.
            infos: Additional information for each game.

        Returns:
            Text commands to be performed (one per observation).
            If episode had ended (e.g. `all(dones)`), the returned
            value is ignored.

        Notes:
            Commands returned for games marked as `done` have no effect.
            The states for finished games are simply copy over until all
            games are done.
        """
        if self.mode == "train" and self.query is not None and self.response is not None:
            reward = torch.tensor(scores[0], dtype=torch.float)
            stats = self.ppo_trainer.step([self.query], [self.response], [reward])
            wandb.log(stats)

        if all(dones):
            self._end_episode(observations, scores, infos)
            return  # Nothing to return.

        if not self._epsiode_has_started:
            self._start_episode(observations, infos)

        # NOTE: this code assumes a batch_size of 1
        assert len(observations) == 1

        obs = observations[0]
        # Remove TextWorld introduction
        obs = obs.split("$$$$ \n\n")[-1]

        self.chat.append({"role": "user", "content": obs})

        # TODO: add initial observation to system prompt
        system = {"role": "system", "content": self.system_prompt}

        prompt = self.tokenizer(
            self.tokenizer.apply_chat_template([system, *self.chat], tokenize=False, add_generation_prompt=True),
            return_tensors="pt"
        ).to("cuda")

        generation_kwargs = {
            "do_sample": True,
            "top_p": 1.0,
            "top_k": 0,
            "pad_token_id": self.tokenizer.eos_token_id,
            "min_length": -1,
            "max_new_tokens": 50,
            # temperature=0.9,
            # repetition_penalty=1.1,
        }

        if self.mode == "train":
            self.query = prompt.input_ids[0]
            responses = self.ppo_trainer.generate(
                self.query,
                batch_size=1,
                return_prompt=False,
                **generation_kwargs
            )
        else:
            responses = self.model.generate(prompt.input_ids, **generation_kwargs)

        assert len(responses) == 1
        self.response = responses[0]

        output = self.tokenizer.decode(self.response, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        output = output.split("[/INST]")[-1].strip()

        if "command: " in output.lower():
            command = output.lower().split("command: ")[-1]
            # HACK: sometimes LLM will output unwanted text after command in new line
            command = command.split("\n")[0]
        else:
            command = "wait"

        # Keep only the actual command part of the response
        self.chat.append({"role": "assistant", "content": f"command: {command}"})

        return [command]
