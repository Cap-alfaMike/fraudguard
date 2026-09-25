"""Políticas de plataforma como código: segurança e confiabilidade dos manifestos."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
K8S = ROOT / "deploy" / "k8s" / "base"


def _load(name):
    return yaml.safe_load((K8S / name).read_text())


@pytest.fixture(scope="module")
def deployment():
    return _load("deployment.yaml")


def test_pod_runs_as_non_root_with_seccomp(deployment):
    sc = deployment["spec"]["template"]["spec"]["securityContext"]
    assert sc["runAsNonRoot"] and sc["runAsUser"] == 10001 and sc["seccompProfile"]["type"] == "RuntimeDefault"


def test_container_hardening(deployment):
    c = deployment["spec"]["template"]["spec"]["containers"][0]
    sc = c["securityContext"]
    assert sc["readOnlyRootFilesystem"] and not sc["allowPrivilegeEscalation"] and sc["capabilities"]["drop"] == ["ALL"]
    assert c["resources"]["requests"]["cpu"] and c["resources"]["limits"]["memory"]


def test_probes_match_api_contract(deployment):
    c = deployment["spec"]["template"]["spec"]["containers"][0]
    assert c["readinessProbe"]["httpGet"]["path"] == "/health"
    assert c["livenessProbe"]["httpGet"]["path"] == "/health/live"  # liveness não depende do modelo


def test_zero_downtime_rollout_and_ha(deployment):
    assert deployment["spec"]["strategy"]["rollingUpdate"]["maxUnavailable"] == 0
    assert deployment["spec"]["replicas"] >= 3
    assert _load("pdb.yaml")["spec"]["minAvailable"] >= 2
    assert _load("hpa.yaml")["spec"]["minReplicas"] >= 3


def test_no_service_account_token(deployment):
    assert deployment["spec"]["template"]["spec"]["automountServiceAccountToken"] is False


def test_network_policy_is_default_deny_style():
    np_ = _load("networkpolicy.yaml")
    assert set(np_["spec"]["policyTypes"]) == {"Ingress", "Egress"}


def test_production_image_not_latest():
    kz = yaml.safe_load((ROOT / "deploy/k8s/overlays/production/kustomization.yaml").read_text())
    assert all(img["newTag"] != "latest" for img in kz["images"])


def test_compose_api_is_hardened():
    api = yaml.safe_load((ROOT / "docker-compose.yml").read_text())["services"]["api"]
    assert api["read_only"] and "ALL" in api["cap_drop"]


def test_prometheus_loads_all_rule_files():
    cfg = yaml.safe_load((ROOT / "ops/prometheus/prometheus.yml").read_text())
    for f in cfg["rule_files"]:
        assert (ROOT / "ops/prometheus" / f).exists()


def test_dockerfile_uses_non_root_user():
    text = (ROOT / "Dockerfile").read_text()
    assert "USER 10001:10001" in text and "HEALTHCHECK" in text
