all: remove prepare

include: .env

.PHONY: delete clean build run exec logs remove prepare

BUILDER = docker
RUNNER = kubectl
gprepare: build push run exec logs

remove: delete clean
	@echo "remove completed!"

build:
	$(BUILDER) build -t $(REPO) .
	@echo "build completed!"

run:
	$(RUNNER) apply -f resources/isvc.yaml
	$(RUNNER) -n $(NAMESPACE) wait --for=condition=Ready pod/netutils --timeout=300s
	$(RUNNER) -n $(NAMESPACE) cp ./scripts/input-v2.json netutils:/
	$(RUNNER) -n $(NAMESPACE) cp ./scripts/test-api.sh netutils:/
	@echo "run completed!"

logs:
	$(RUNNER) -n $(NAMESPACE) logs $$($(RUNNER) -n $(NAMESPACE) get po | grep 'predictor' | awk '{print $$1}')

exec:
	$(RUNNER) -n $(NAMESPACE) wait --for=condition=Ready isvc/test --timeout=300s
	$(RUNNER) -n $(NAMESPACE) exec -it netutils -- bash /test-api.sh
	@echo "\nexec completed!"

push:
	$(BUILDER) push $(REPO)
	@echo "push completed!"

delete:
	$(RUNNER) -n $(NAMESPACE) delete -f resources/isvc.yaml
	$(RUNNER) -n $(NAMESPACE) wait --for=condition=Ready pods --all --timeout=300s
	@echo "delete completed!"

clean:
	$(BUILDER) rmi $(REPO)
	bash ./scripts/harbor-cmd.sh
	@echo "clean completed!"
