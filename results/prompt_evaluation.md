# Intent evaluation (test split, provider=ollama, model=llama3.2:3b)

| method     |   intent_accuracy |   target_accuracy_apps_folders |   valid_json_rate |   samples |   seconds |
|:-----------|------------------:|-------------------------------:|------------------:|----------:|----------:|
| rules      |              93.5 |                            100 |               100 |        31 |       0   |
| classifier |              93.5 |                             90 |               100 |        31 |       1.8 |
| llm-v1     |              74.2 |                             90 |               100 |        31 |      38.9 |
| llm-v2     |              93.5 |                            100 |               100 |        31 |      30.6 |
| llm-v3     |              96.8 |                            100 |               100 |        31 |      53.4 |
| llm-v4     |              93.5 |                            100 |               100 |        31 |      55.4 |