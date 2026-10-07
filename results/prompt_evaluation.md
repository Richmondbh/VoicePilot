# Intent evaluation (test split, provider=ollama, model=llama3.2:3b)

| method     |   intent_accuracy |   target_accuracy_apps_folders |   valid_json_rate |   samples |   seconds |
|:-----------|------------------:|-------------------------------:|------------------:|----------:|----------:|
| rules      |              96.8 |                           90.9 |               100 |        31 |       0   |
| classifier |              90.3 |                          100   |               100 |        31 |      14   |
| llm-v1     |              74.2 |                          100   |               100 |        31 |      40.8 |
| llm-v2     |              90.3 |                          100   |               100 |        31 |      29.3 |
| llm-v3     |              96.8 |                           90.9 |               100 |        31 |      50.2 |