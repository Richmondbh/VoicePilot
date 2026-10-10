# Intent evaluation (test split, provider=ollama, model=llama3.2:3b)

| method     |   intent_accuracy |   target_accuracy_apps_folders |   valid_json_rate |   samples |   seconds |
|:-----------|------------------:|-------------------------------:|------------------:|----------:|----------:|
| rules      |              93.8 |                           90.9 |               100 |        32 |       0   |
| classifier |              90.6 |                           90.9 |               100 |        32 |       1.4 |
| llm-v1     |              81.2 |                          100   |               100 |        32 |      37.1 |
| llm-v2     |              93.8 |                          100   |               100 |        32 |      27.6 |
| llm-v3     |              96.9 |                           90.9 |               100 |        32 |      45.3 |
| llm-v4     |              90.6 |                           90.9 |               100 |        32 |      47.9 |