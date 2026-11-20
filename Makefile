.DEFAULT_GOAL := help
.PHONY: help demo

help: ## Show available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-10s %s\n", $$1, $$2}'

demo: ## Run the whole app locally, then open http://localhost:3000
	./scripts/demo.sh
