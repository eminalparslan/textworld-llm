from typing import List, Dict, Any, Optional

from textworld import EnvInfos

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer
import transformers
import torch

from collections import deque



class CustomAgent:
    """ Template agent for the TextWorld competition. """

    def __init__(self) -> None:
        self._initialized = False
        self._epsiode_has_started = False

        # self.model = LlamaForCausalLM.from_pretrained("./submission_llm/llama-2-7b/consolidated.00.pth")
        # self.model =

        self.system_prompt = "You are an agent playing a text-based game. You will be given observations and respond to them with commands. Here are some example commands: 'go west', 'inventory', 'drop teacup', 'examine broom', 'close type 2 box', 'open door', 'insert pencil into type 1 box', 'look'. Not all commands will work, but there are many others that you can try beyond these examples. When responding, first reason about the game state to decide the best action and then say 'command: <your command>'. Do not say anything other than the command you chose."
        self.chat = deque(maxlen=11)
        #self.chat.append({"role": "system", "content": self.system_prompt},)

        model = "meta-llama/Llama-2-7b-chat-hf"
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            torch_dtype=torch.float16,
            device_map="auto",
        )

    def train(self) -> None:
        """ Tell the agent it is in training mode. """
        raise NotImplementedError

    def eval(self) -> None:
        """ Tell the agent it is in evaluation mode. """
        pass  # [You can insert code here.]

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

        # [You can insert code here.]

    def _end_episode(self, obs: List[str], scores: List[int], infos: Dict[str, List[Any]]) -> None:
        """
        Tell the agent the episode has terminated.

        Arguments:
            obs: Previous command's feedback for each game.
            score: The score obtained so far for each game.
            infos: Additional information for each game.
        """
        self._epsiode_has_started = False

        print(f"{scores=}")

    def act(self, obs: List[str], scores: List[int], dones: List[bool], infos: Dict[str, List[Any]]) -> Optional[List[str]]:
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
        if all(dones):
            self._end_episode(obs, scores, infos)
            return  # Nothing to return.

        if not self._epsiode_has_started:
            self._start_episode(obs, infos)

        obs = [o.split("$$$$ \n\n")[-1] for o in obs]

        # FIXME: assuming batch size of 1
        self.chat.append({"role": "user", "content": obs[0]})

        system = {"role": "system", "content": self.system_prompt}

        sequences = self.pipeline(
            self.tokenizer.apply_chat_template([system, *self.chat], tokenize=False, add_generation_prompt=True),
            # https://arc.net/l/quote/fqgyfjis
            do_sample=True,
            top_p=0.6,
            top_k=0,
            temperature=0.9,
            num_return_sequences=1,
            eos_token_id=self.tokenizer.eos_token_id,
            batch_size=len(obs),
            max_new_tokens = 50,
        )

        if not sequences:
            raise Exception("Model has no output")

        commands = [s["generated_text"].split("[/INST]")[-1].strip() for s in sequences]

        # FIXME: assuming batch size of 1
        self.chat.append({"role": "assistant", "content": commands[0]})

        result = []
        for c in commands:
            if "command: " in c.lower():
                idx = c.lower().index("command: ")
                idx += len("command: ")
                result.append(c[idx:])
            else:
                result.append("wait")

        print(f"{result=}")

        return result
