## Evaluation

For evaluating the agents we create a list of topics stored in `topics.txt`.

The agents are then run, passing the topic as input and we evaluate how many times the agent is able to guess the topic.

### Metrics

- **Success Rate**: Percentage of topics guessed correctly.
- **Average Questions**: Average number of questions asked to guess the topic.
- **Average Time**: Average time taken to guess the topic.
- **Error Rate**: Percentage of topics that caused an error.

### Parallel Execution

Currently, the evaluation is not parallelized.

However, the evaluation can be parallelized by bringing about parallel execution at the level of passing `test_topics` to the agents.
