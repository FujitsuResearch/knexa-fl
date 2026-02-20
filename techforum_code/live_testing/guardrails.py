#!/usr/bin/env python3
"""
Rule-based + optional LLM-based guardrails for live testing.

Design goals:
- High precision: block only clearly harmful/malicious content or requests.
- Low false positives: allow benign, policy/awareness, detection/mitigation contexts.
- Explanations: return concise reason and category for any block.

Usage:
- Call check_input(prompt, ...) before generation.
- Call check_output(text, ...) after generation.
"""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional, Dict, Any


NEUTRAL_MESSAGE = (
    "This content is blocked for safety. I can offer high-level, non-harmful guidance instead."
)


@dataclass
class GuardResult:
    label: str  # 'safe' | 'blocked' | 'fallback'
    reason: str
    category: Optional[str] = None


def _compile_patterns():
    # Intent cues (imperative, enabling misuse)
    intent = r"(write|craft|compose|generate|create|build|make|design|produce|provide|give|share|supply|teach|explain how to|step[- ]?by[- ]?step|instructions? for|bypass|evad(?:e|ing)|circumvent|spoof|impersonat|steal|hack|exploit|undetectable|no\s?logs|disable\s?logs)"
    # Safety context allow-list cues (avoid false positives)
    allow_ctx = r"(awareness|detect(?:ion)?|mitigat(?:e|ion)|prevent(?:ion)?|policy|compliance|best practices|training|education|guidance|how to avoid|how to detect|defen[cs]e)"

    rules = [
        # Cybersecurity misuse
        dict(
            name="cyber_misuse",
            pattern=re.compile(rf"\b{intent}\b.*\b(malware|ransomware|keylogger|backdoor|rootkit|ddos|d\W?dos|c2|command and control|botnet|reverse shell|payload|shellcode|cryptominer|rat|remote admin tool|data exfiltration|exfiltrate)\b", re.I),
            except_when=re.compile(rf"\b{allow_ctx}\b", re.I),
            explanation="Requests assisting cyberattack tooling or code are disallowed.",
        ),
        # Phishing/social engineering
        dict(
            name="social_engineering",
            pattern=re.compile(rf"\b{intent}\b.*\b(phish(?:ing)?|spear\W?phish|whale\W?phish|vishing|smishing|credential\s?(harvest|stuffing|spray|spraying)|mfa\s?fatigue|business\s?email\s?compromise|bec|pretext(?:ing)?|spoof(?:ed)?\s?(domain|email)|impersonat)\b", re.I),
            except_when=re.compile(rf"\b{allow_ctx}\b", re.I),
            explanation="Requests enabling social engineering or credential theft are disallowed.",
        ),
        # Security bypass
        dict(
            name="security_bypass",
            pattern=re.compile(rf"\b{intent}\b.*\b(2fa|mfa|rate\s?limit|firewall|edr|xdr|siem|ids|ips|detection|guardrail|moderation|filter|sandbox|waf|sso|okta|authn|authz|auth|authorization|authentication|logging)\b", re.I),
            except_when=re.compile(rf"\b{allow_ctx}\b", re.I),
            explanation="Requests to bypass or evade security controls are disallowed.",
        ),
        # Web exploits
        dict(
            name="web_exploit",
            pattern=re.compile(rf"\b{intent}\b.*\b(sql\s?injection|sqli|xss|csrf|ssti|rce|lfi|rfi|xxe|object\s?injection|deserialization\s?attack|buffer\s?overflow)\b", re.I),
            except_when=re.compile(rf"\b{allow_ctx}\b", re.I),
            explanation="Requests enabling exploitation techniques are disallowed.",
        ),
        # Financial fraud/PII harvesting
        dict(
            name="fraud_pii",
            pattern=re.compile(rf"\b{intent}\b.*\b(ssn|social\s?security|credit\s?card|cvv|otp|one[- ]?time\s?pass|bank\s?account|routing\s?number|iban|swift|pin|kyc\s?bypass|money\s?mule|carding|skimmer)\b", re.I),
            except_when=re.compile(rf"\b{allow_ctx}\b", re.I),
            explanation="Requests aimed at obtaining sensitive personal/financial data are disallowed.",
        ),
        # Physical harm/illegal goods
        dict(
            name="physical_harm",
            pattern=re.compile(rf"\b{intent}\b.*\b(weapon|ghost\s?gun|3d\s?printed\s?gun|silencer|bomb|explosive|tatp|hmtd|fentanyl|meth(?:amphetamine)?|illegal\s?drug|narcotic|poison|kill|assassinat|harm\s?someone)\b", re.I),
            except_when=re.compile(rf"\b{allow_ctx}\b", re.I),
            explanation="Requests promoting physical harm or illegal activity are disallowed.",
        ),
        # Jailbreak/policy evasion
        dict(
            name="policy_evasion",
            pattern=re.compile(r"\b(jailbreak|ignore previous instructions|act without restrictions|no safety|no guardrails)\b", re.I),
            except_when=None,
            explanation="Attempts to disable safety policies are disallowed.",
        ),
    ]
    return rules


RULES = _compile_patterns()


def _apply_rules(text: str) -> GuardResult:
    lowered = text.lower()
    for r in RULES:
        if r["pattern"].search(lowered):
            ex = r.get("except_when")
            if ex and ex.search(lowered):
                continue
            return GuardResult(label="blocked", reason=r["explanation"], category=r["name"])
    return GuardResult(label="safe", reason="pass")


def _llamaguard(text: str, model_id: str = "meta-llama/Llama-Guard-3-8B", device: Optional[str] = None) -> GuardResult:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        trust_remote_code = True
        device_map = "auto" if device in (None, "auto") else {"": device}
        quant_kwargs = {}
        try:
            from transformers import BitsAndBytesConfig
            quant_kwargs = {
                "load_in_4bit": True,
                "quantization_config": BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                ),
            }
        except Exception:
            pass
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            device_map=device_map,
            torch_dtype=torch.float16,
            **quant_kwargs,
        )
        prompt = f"[User]: {text}\n[Assistant]:"
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=16, temperature=0.0, do_sample=False)
        gen = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        l = gen.lower()
        if any(x in l for x in ("unsafe", "not allowed", "refuse", "deny", "blocked")):
            return GuardResult(label="blocked", reason=gen[:240], category="llamaguard")
        return GuardResult(label="safe", reason=gen[:240] or "pass")
    except Exception as e:
        return GuardResult(label="fallback", reason=f"llamaguard_unavailable: {e}")


def check_input(text: str, use_llamaguard: bool = False, safety_model_id: str = "meta-llama/Llama-Guard-3-8B", device: Optional[str] = None) -> GuardResult:
    # Try LlamaGuard if enabled; otherwise use rules
    if use_llamaguard:
        res = _llamaguard(text, model_id=safety_model_id, device=device)
        if res.label != "fallback":
            return res
    return _apply_rules(text)


def check_output(text: str, use_llamaguard: bool = False, safety_model_id: str = "meta-llama/Llama-Guard-3-8B", device: Optional[str] = None) -> GuardResult:
    # Post-generation safety; same cascade
    if use_llamaguard:
        res = _llamaguard(text, model_id=safety_model_id, device=device)
        if res.label != "fallback":
            return res
    return _apply_rules(text)


def block_message(reason: str, category: Optional[str]) -> str:
    cat = f"[{category}] " if category else ""
    return f"Blocked by guardrails {cat}- {reason}\n\n{NEUTRAL_MESSAGE}"
