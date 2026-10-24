.PHONY: test proto lint benchmark run

test:
	PYTHONPATH=src python3 -m pytest tests/ -v

benchmark:
	PYTHONPATH=src python3 benchmarks/run_qwen_comparison.py

proto:
	python3 -m grpc_tools.protoc -Iproto --python_out=src/paxlet_nl_dsl_llm --grpc_python_out=src/paxlet_nl_dsl_llm proto/paxlet/nl_dsl_llm/v1/runtime.proto

run:
	PYTHONPATH=src python3 -m paxlet_nl_dsl_llm.server
