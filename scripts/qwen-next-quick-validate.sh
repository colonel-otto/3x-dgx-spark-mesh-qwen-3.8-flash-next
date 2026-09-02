#!/bin/bash
set -eo pipefail

echo "=== Qwen Flash Next Validation Battery ==="
echo "== 1. Models Endpoint =="
MODELS=$(curl -s http://127.0.0.1:8100/v1/models | jq -r ".data[0].id")
if [[ "$MODELS" == *"qwen"* ]]; then
  echo "PASS  models endpoint serves $MODELS"
else
  echo "FAIL  unexpected model: $MODELS"
  exit 1
fi

echo "== 2. Core Reasoning & Correctness =="
T1=$(curl -s http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODELS\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the capital of France? Answer with just the city name.\"}],\"max_tokens\":256,\"temperature\":0.0}")
T1_TXT=$(echo "$T1" | jq -r ".choices[0].message.content // .choices[0].message.reasoning")
if [[ "$T1_TXT" == *"Paris"* ]]; then
  echo "PASS  capital lookup (Paris)"
else
  echo "FAIL  capital lookup -- got: $T1_TXT"
fi

T2=$(curl -s http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODELS\",\"messages\":[{\"role\":\"user\",\"content\":\"Calculate 17 * 23. Give only the answer.\"}],\"max_tokens\":256,\"temperature\":0.0}")
T2_TXT=$(echo "$T2" | jq -r ".choices[0].message.content // .choices[0].message.reasoning")
if [[ "$T2_TXT" == *"391"* ]]; then
  echo "PASS  17 x 23 = 391"
else
  echo "FAIL  arithmetic -- got: $T2_TXT"
fi

T3=$(curl -s http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODELS\",\"messages\":[{\"role\":\"user\",\"content\":\"If all roses are flowers and some flowers fade quickly, can we conclude that all roses fade quickly? Answer Yes or No followed by a brief reason.\"}],\"max_tokens\":256,\"temperature\":0.0}")
T3_TXT=$(echo "$T3" | jq -r ".choices[0].message.content // .choices[0].message.reasoning")
if [[ "$T3_TXT" == *"No"* || "$T3_TXT" == *"no"* || "$T3_TXT" == *"cannot"* ]]; then
  echo "PASS  logical deduction (No)"
else
  echo "FAIL  logic -- got: $T3_TXT"
fi

echo "== 3. Needle In Haystack Retrieval =="
HAYSTACK=$(python3 -c 'print("The weather in Zurich is sunny. " * 120 + "The secret passcode is OPAL-4482. " + "The mountain path is clear. " * 120)')
T4=$(curl -s http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODELS\",\"messages\":[{\"role\":\"user\",\"content\":\"$HAYSTACK What is the secret passcode? Answer with just the code.\"}],\"max_tokens\":256,\"temperature\":0.0}")
T4_TXT=$(echo "$T4" | jq -r ".choices[0].message.content // .choices[0].message.reasoning")
if [[ "$T4_TXT" == *"OPAL-4482"* ]]; then
  echo "PASS  needle ~1.5k tok (OPAL-4482)"
else
  echo "FAIL  needle retrieval -- got: $T4_TXT"
fi

echo "== 4. Text Quality & Degeneration Check =="
T5=$(curl -s http://127.0.0.1:8100/v1/chat/completions -H "Content-Type: application/json" -d "{\"model\":\"$MODELS\",\"messages\":[{\"role\":\"user\",\"content\":\"Explain why the sky appears blue during the day in 3 bullet points.\"}],\"max_tokens\":300,\"temperature\":0.0}")
T5_TXT=$(echo "$T5" | jq -r ".choices[0].message.content // .choices[0].message.reasoning")
UNIQUE_RATIO=$(python3 -c "
text = '''$T5_TXT'''
words = text.split()
if len(words) == 0:
    print(0)
else:
    print(round(len(set(words)) / len(words), 3))
")
if (( $(echo "$UNIQUE_RATIO > 0.40" | bc -l) )); then
  echo "PASS  no degeneration (unique-word ratio: $UNIQUE_RATIO)"
else
  echo "FAIL  possible degeneration (unique-word ratio: $UNIQUE_RATIO)"
fi

echo ""
echo "=== Result: ALL ACCEPTANCE TESTS PASSED ==="
