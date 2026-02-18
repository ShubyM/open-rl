raw stream of thoughts:

RL typically involves a training loop where we feed data which is organized in batches. So if there are N batches, we will run a N steps of RL loop.

The whole process begins with a pre-trained model and a dataset. 

Now key thing is that the dataset is just initial prompts. It is not complete yet. RL involves using the model that is being trained to generate the rest of the data for the next iteration of training.

So in each iteration of RL, the data generated in the next iteration depends on the model updated in the previous iteration.

So input to RL:
 - base model
 - dataset of prompts

## Life of a single row in a batch
 - taking a prompt from the dataset (or data source) (ex. "what is the weather like today?")
 - Use current model to generate a response. This may involve multiple steps of generation. For example, if the prompt is "what is the weather like today?", the model may first generate "weather depends on location? and it may suggest to use a tool to get the weather or ask for the location". In the second step, model will be provided with the location (either by invoking the tool or by providing it as part of the prompt). Then model will generate second response that will involve invoking the tool to get the weather for the location. Then model will be provided with the weather information and it will generate the final response. so we will have a mult-turn conversation that have tool-calls and tool-responses. This multi-turn conversation represents a complete row of data. In a batch of N prompts, we will have N such conversations. These multi-turn conversations are also referred to as trajectories or episodes. The multi-turn execution is also called agentic loop or an agent completing an episode.
  - Note the tool-calls may require interacting with external systems. These external systems are referred to as environment. For a coding domain, the environment is a sandbox with code interpreter with access to file-system and ability to run code and see the output.
 - Since model that is being trained is still learning, so its multi-turn response may not be desired or accurate yet. So we need a way to grade the episode. The result of grading will be a reward (positive or negative) or a preference between two episodes. Now grading process may involve human annotators or it may involve another model (called reward model) that is trained to grade the episodes or testing out the output of episodes in the environment and getting a reward based on the output. For example, in a coding domain, compiling the code and running test cases may give a reward. If the code compiles and passes all test cases, we give a positive reward. If the code fails to compile or fails test cases, we give a negative reward. Also, reward may be different dimensions. For example, code that is syntatically correct (compiles successfully) can be given some points, code the passes the test cases can be given some points and so on.
  - So a completed row of data will include: (initial prompt, multi-turn conversation (trajectory), reward/preference)

So preparing a batch of data, will involve running N episodes using the current model. This process is also called Generation. 

## Training Loop

We start with a base model M1 and a dataset of prompts.

In the first iteration of RL, we will use M1 to generate a batch of data using the agentic loop, environment and reward.
Train M1 on this batch of data to get M2.
The, we will use M2 to generate a new batch of data using the agentic loop, environment and reward.
Train M2 on this batch of data to get M3.
We repeat this process for K iterations.

## Weight Copying


There is another key aspects that I forgot to specify earlier. 
Say a training step is completed for M1 and M2 is produced. Now typically training infra is separate from inference infra, so generation of responses for new batch need to happen from M2. And for that, weights of M2 model need to be synced to inference infra so that next batch uses the current model.

So two things I am wondering how to represent these aspects. Should we introduce "Weight Sync" step somewhere. Also How do we represent inference component like we represent training component. Before updating, pl. bounce ideas with me here.

## Throughput

There is this aspect about the RL components that I am not so sure how to represent or may be we create a type of diagram for that.

So as a user who is post training a model. they care about the throughput for their whole RL training. Through depends on many factors:

 - Training loop can run only as fast the underlying GPUs and data batches availability
 - Data batches availability depnds on generation effeciency, tool calls/response, environment, grading 

Some of these can be solves by the infrastructure provider such as powerful training infra (GPUs/TPUs), inference infra (GPUs/TPUs), but then other aspects depend on the user  (multi-turn, tool-calls, environment, grader) etc.

How do we capture that.
