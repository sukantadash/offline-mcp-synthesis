# offline-mcp-synthesis

Generate simulated tool-calling training conversations from sample transcripts, seed queries, a system prompt, and a tool schema. A teacher model writes the new questions and the tool results. This run does not start the sample MCP server, register tools with the model, or call an agent.

The behavior is specified in [DESIGN.md](DESIGN.md). The Order Ops files in this folder are one server pack. Replacing the pack is how you point the same flows at another MCP server.

Two more scripts build training data from executed tool calls instead. `stateful_env.py` runs an agent against `mcp_server.py` with a fresh state per scenario (section 7). `traces_to_training.py` rebuilds training conversations from recorded MLflow traces (section 8).

## 1. Create a Python 3.12 environment

From this folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

`sdg_hub` is installed from the package name in `requirements.txt`. If you are working from the local SDG Hub checkout, install that checkout instead:

```bash
python -m pip install -e ../../sdg_hub
python -m pip install pandas python-dotenv jsonschema fastmcp mlflow
```

## 2. Configure the teacher

Create `.env` in this folder:

```dotenv
TEACHER_MODEL_NAME=openai/your-served-model
TEACHER_MODEL_URL=https://your-model-endpoint/v1
TEACHER_API_KEY=your-key
```

`TEACHER_MODEL_NAME` uses LiteLLM's provider/model form. For an OpenAI-compatible endpoint that is `openai/<served-model-name>`. `MODEL`, `API_BASE`, `API_KEY`, and `OPENAI_API_KEY` are accepted as fallbacks.

## 3. Review the Order Ops pack

These files are the inputs. The flows do not infer seeds from the samples.

| File | Role |
| --- | --- |
| `input/examples.jsonl` | Sample tool-calling conversations, including one inconsistent shipment answer the teacher is told not to copy |
| `input/seeds.json` | Seed queries |
| `input/system_prompt.txt` | System prompt copied into every generated transcript |
| `input/tool_schema.json` | OpenAI function schemas for `list_orders`, `get_order`, `check_inventory`, and `create_shipment` |
| `generation/prompts/synthesize.yaml` | Domain rules for writing a conversation |
| `review/prompts/review.yaml` | Domain rules for the consistency review |
| `input/scenario_states.json` | Orders and inventory per scenario, used by `stateful_env.py` |

`mcp_server.py` is a runnable FastMCP server with the same catalog. Synthesis does not connect to it; `stateful_env.py` starts it once per scenario (section 7). Start it yourself when you want to call the tools:

```bash
python mcp_server.py
```

It listens on `http://127.0.0.1:8008/mcp`. `PORT` and `HOST` override the bind address.

## 4. Run the tests

No test calls a model endpoint. The synthesis tests mock the teacher and exercise the Order Ops tool functions in-process. The stateful environment tests start the Order Ops server as a subprocess and call it over MCP. The trace tests write MLflow traces to a temporary store.

```bash
python -m unittest discover -s tests -t . -v
```

## 5. Generate conversations

```bash
python offline_mcp_synthesis.py
```

The command reads `.env` and the files in `input/`, then writes:

- `output/tool_calling_dataset_generated.jsonl` for the accepted records followed by the rejected records
- `output/tool_calling_dataset_synthetic.jsonl` for rows that pass the filter
- `output/tool_calling_dataset_synthetic.rejected.jsonl` for rows rejected during structural checks, review parsing, or quality filtering

A run with zero accepted rows still exits successfully and still writes the rejection file. Read both files before using them for training. Tool results are simulated. The review score is a consistency heuristic.

Optional environment overrides:

| Variable | Default |
| --- | --- |
| `VARIANTS_PER_SEED` | `2` |
| `QUALITY_CHECK` | `true` |
| `MIN_QUALITY_SCORE` | `0.8` |
| `EXAMPLES_PATH` | `input/examples.jsonl` |
| `SEEDS_PATH` | `input/seeds.json` |
| `SYSTEM_PROMPT_PATH` | `input/system_prompt.txt` |
| `TOOL_SCHEMA_PATH` | `input/tool_schema.json` |
| `OUTPUT_PATH` | `output/tool_calling_dataset_synthetic.jsonl` |

Set `QUALITY_CHECK=false` to keep every row that passes structural validation and skip the review flow.

## 6. Use another MCP server

Replace the pack together. Leave `generation/flow.yaml`, `review/flow.yaml`, and `filter/flow.yaml` as they are.

1. Put the new function schemas in `input/tool_schema.json` (or point `TOOL_SCHEMA_PATH` at another file).
2. Replace `input/examples.jsonl` with transcripts that use those tools.
3. Replace `input/seeds.json` and `input/system_prompt.txt`.
4. Rewrite the domain sentences in `generation/prompts/synthesize.yaml` and `review/prompts/review.yaml`.
5. Replace `mcp_server.py` if you want a live server for inspection. The synthesizer still will not call it; `stateful_env.py` does.

Then repeat steps 4 and 5.

## 7. Run an agent against a stateful mock environment

`stateful_env.py` produces conversations whose tool results come from executing the tools. For every scenario in `input/scenario_states.json` and every seed query, it:

1. writes the scenario's orders and inventory to a state file,
2. starts `mcp_server.py` on a free port with `ORDER_OPS_STATE` pointing at that file,
3. runs the teacher model as the agent through SDG Hub's `MCPAgentBlock`,
4. reads the final state back and rejects the run unless every change is explained by a successful `create_shipment` result in the transcript.

```bash
python stateful_env.py
```

Outputs:

- `output/stateful_env_dataset.jsonl`: accepted rows with `messages` (assistant `tool_calls` and `tool` results), `tools`, `scenario`, `seed_query`, `initial_state`, `final_state` and `state_changes`
- `output/stateful_env_dataset.rejected.jsonl`: rows that failed structural validation or the state check
- `output/stateful_env_logs/`: one server log per run

Each run is also traced to MLflow (`output/mlflow.db`, experiment `order-ops-stateful-env`). `mlflow.litellm.autolog()` records the model calls, and each MCP tool call is recorded as a TOOL span. Use `--no-tracing` to skip tracing, or `--mlflow-tracking-uri` to send traces elsewhere.

To use another server, have it load a state file at startup and write it back after every change, and replace `expected_state` in `stateful_env.py` with that server's write rules.

## 8. Build training data from recorded traces

`traces_to_training.py` turns recorded agent traces into the same training format. It reads an MLflow tracking store, or exported trace JSON files:

```bash
python traces_to_training.py                               # output/mlflow.db, experiment order-ops-stateful-env
python traces_to_training.py --trace-files traces/*.json   # exported traces
python traces_to_training.py --export-dir output/traces    # also save each trace as JSON
```

Each conversation is rebuilt from the trace's last LLM span: its input messages plus its response. Traces without LLM spans are rebuilt from the root span and the TOOL spans; `--source tool-spans` forces this. A conversation is kept only if it passes the structural checks used elsewhere in this repo and every tool call and result matches a recorded TOOL span. Duplicate conversations are dropped.

Outputs:

- `output/traces_training.jsonl`: accepted rows with `messages`, `tools`, `source_question` and `trace_id`
- `output/traces_training.rejected.jsonl`: rejected rows with the reason

Rebuilding from TOOL spans expects the root span to record `system_prompt` and `query` as inputs and `answer` as output, as `stateful_env.py` does. Traces from another agent runtime need their span layout mapped the same way.
