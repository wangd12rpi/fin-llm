# Evaluation Results

## Accuracy

| Model | fiqa | fpb | headline | ner | nwgi | tfns |
|------| --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-3.1-8B-Instruct--1bits-r-1 | 0.4545 | 0.7071 | 0.4537 | 0.4863 | 0.4598 | 0.7010 |
| meta-llama/Llama-3.1-8B-Instruct-4bits-r4 | 0.7236 | 0.8647 | - | - | - | 0.8798 |
| meta-llama/Llama-3.1-8B-Instruct-4bits-r8 | - | - | - | - | - | - |
| meta-llama/Llama-3.1-8B-Instruct-8bits-r4 | - | - | - | - | - | - |
| meta-llama/Llama-3.1-8B-Instruct-8bits-r8 | 0.8364 | 0.8317 | 0.8350 | 0.9101 | - | 0.8338 |


## F1 Score

| Model | fiqa | fpb | headline | ner | nwgi | tfns |
|------| --- | --- | --- | --- | --- | --- |
| meta-llama/Llama-3.1-8B-Instruct--1bits-r-1| 0.5416 | 0.6973 | 0.5570 | 0.5636 | 0.4064 | 0.6818 |
| meta-llama/Llama-3.1-8B-Instruct-4bits-r4| 0.7742 | 0.8612 | - | - | - | 0.8797 |
| meta-llama/Llama-3.1-8B-Instruct-4bits-r8| - | - | - | - | - | - |
| meta-llama/Llama-3.1-8B-Instruct-8bits-r4| - | - | - | - | - | - |
| meta-llama/Llama-3.1-8B-Instruct-8bits-r8| 0.8428 | 0.8331 | 0.8417 | 0.9090 | - | 0.8372 |
