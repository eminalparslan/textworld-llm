- [x] Experiment with TextWorld
- [ ] Run LSTM-DQN example
    - [ ] Use LLM embeddings as the state space so it also has baseline commonsense
      knowledge (as recommended by Caroline)
      - Look at the paper that used BERT
    - [ ] The LSTM infuses temporal information. How do I get this in the LLM
      with RL fine-tuning?
      - It uses a replay buffer. Maybe I can just keep such a buffer in the LLM context?
- [ ] Use HF TRL and PEFT libraries to fine-tune
- [ ] Experiments
  - Just LLM
  - LLM w/ RL fine-tuning
  - LLM embeddings w/ DRRN and/or LSTM-DQN
  - Just LSTM-DQN

Question: How can RL direct a LM to environments with languistic state/action spaces.
