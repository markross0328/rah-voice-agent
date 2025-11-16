run:
	docker compose -f config/compose.yaml up --build

test:
	pytest -q

repro CALL_ID?=latest
repro:
	python -m offline.repro_harness.run_repro --call-id $(CALL_ID)
