#!/bin/sh
set -eu
cd "$(dirname "$0")/../.."
# Only this dedicated cluster. Results, unrelated clusters/images and Docker volumes remain.
kind delete cluster --name mlops-hw11 --kubeconfig notes/work/hw11/kubeconfig
# Optional after checking its use: docker image rm planner-hw11:v3 planner-hw11:v1
