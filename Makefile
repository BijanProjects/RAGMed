.DEFAULT_GOAL := help
.PHONY: help demo infra-up infra-down

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

demo: ## Run the whole app locally, then open http://localhost:3000
	./scripts/demo.sh

infra-up: ## Start local Postgres and Valkey (Docker)
	docker compose -f infra/compose/dev.yml up -d --wait

infra-down: ## Stop local Postgres and Valkey (data is kept)
	docker compose -f infra/compose/dev.yml down
