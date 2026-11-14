#!/bin/sh
set -eu
# Prerequisites: Docker, kubectl, kind v0.30.0, Helm v3.19.0; one private local context.
cd "$(dirname "$0")/../.."
mkdir -p notes/work/hw11
kind create cluster --name mlops-hw11 --image kindest/node:v1.34.0 --kubeconfig notes/work/hw11/kubeconfig --wait 120s
# Limit the build context via .dockerignore; no model weights enter this image.
docker build -f Dockerfile.k8s -t planner-hw11:v2 .
kind load docker-image planner-hw11:v2 --name mlops-hw11
helm upgrade --install planner infra/helm/planner --kubeconfig notes/work/hw11/kubeconfig -n mlops-hw11 --create-namespace
kubectl --kubeconfig notes/work/hw11/kubeconfig apply -f infra/monitoring/stack.yaml -f infra/clients.json -f infra/monitoring/metrics-server.yaml
kubectl --kubeconfig notes/work/hw11/kubeconfig -n kube-system patch deploy metrics-server --type=json -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"},{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--metric-resolution=15s"}]'
kubectl --kubeconfig notes/work/hw11/kubeconfig -n mlops-hw11 rollout status deploy/planner --timeout=150s
