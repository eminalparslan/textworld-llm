- [x] Experiment with TextWorld
- [x] Run LSTM-DQN example
    - [ ] Use LLM embeddings as the state space so it also has baseline commonsense
      knowledge (as recommended by Caroline)
      - Look at the paper that used BERT
    - [ ] The LSTM infuses temporal information. How do I get this in the LLM
      with RL fine-tuning?
      - It uses a replay buffer. Maybe I can just keep such a buffer in the LLM context?
- [ ] Use HF TRL and PEFT libraries to fine-tune
- [ ] Experiments
  - [x] Just LSTM-DQN
  - [x] Just LLM
    - [ ] Try out different handicaps
      - Right now it only uses observations and ignores the infos dict
  - [ ] LLM w/ RL fine-tuning
  - [ ] LLM embeddings w/ DRRN and/or LSTM-DQN

Question: How can RL direct a LM to environments with linguistic state/action spaces.

TBGs have language action and state space. Policy needs to map from state to action. LLMs do this. They will have some base common knowledge. Test with prompting. They can be fine tuned with the reward signal. Like RLHF but for different goal. Compare with traditional techniques imbued with embedding knowledge or BERT?

How handicapped can we go with each technique?

How to use context of LLM? Storing history is cheating? LSTM means model stores state about past. Advantage of GPT is that it can attend to its context window.

Generalizability? Increases complexity and see how different methods scale.
