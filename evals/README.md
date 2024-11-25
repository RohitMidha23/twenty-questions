## Evaluation

For evaluating the agents we create a list of topics stored in `topics.txt`.

The agents are then run, passing the topic as input and we evaluate how many times the agent is able to guess the topic.

### Metrics

- **Success Rate**: Percentage of topics guessed correctly.
- **Average Questions**: Average number of questions asked to guess the topic.
- **Average Time**: Average time taken to guess the topic.
- **Error Rate**: Percentage of topics that caused an error.

### Parallel Execution

The evaluation framework uses parallel execution to efficiently test multiple topics simultaneously. This is implemented using Python's `ThreadPoolExecutor`.

Key aspects of the parallel execution:

1. Each game/topic is run in a separate thread to maximize throughput
2. Progress is tracked using `tqdm` to show completion status
3. Results are collected asynchronously as games complete