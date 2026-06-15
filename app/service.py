from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parents[1]

import sys

sys.path.append(str(ROOT / "src"))

from risk_nlp.candidates import extract_candidate_entities, filter_entities_by_candidates, normalize_entity
from risk_nlp.database import RiskDatabase
from risk_nlp.risk_events import build_risk_events
from risk_nlp.schema import SYSTEM_PROMPT, build_user_prompt, parse_model_output


class RiskWarningService:
    def __init__(
        self,
        *,
        encoder_path: str = "outputs/encoder/valuesimplex-ai-lab__FinBERT2-large/checkpoint-final",
        llm_base: str = "/data1/yangjuhao/models/Qwen3-8B-ModelScope",
        adapter_path: str = "outputs/qwen3-8b-qlora-main/adapter",
        db_path: str = "outputs/risk_system/risk_system.db",
        encoder_threshold: float = 0.5,
    ) -> None:
        self.encoder_path = ROOT / encoder_path
        self.llm_base = llm_base
        self.adapter_path = ROOT / adapter_path
        self.encoder_threshold = encoder_threshold
        self.db = RiskDatabase(ROOT / db_path)
        self.encoder_tokenizer = None
        self.encoder_model = None
        self.llm_tokenizer = None
        self.llm_model = None

    def load_encoder(self) -> None:
        if self.encoder_model is not None:
            return
        self.encoder_tokenizer = AutoTokenizer.from_pretrained(self.encoder_path, use_fast=True)
        self.encoder_model = AutoModelForSequenceClassification.from_pretrained(self.encoder_path)
        if torch.cuda.is_available():
            self.encoder_model = self.encoder_model.cuda()
        self.encoder_model.eval()

    def load_llm(self) -> None:
        if self.llm_model is not None:
            return
        self.llm_tokenizer = AutoTokenizer.from_pretrained(self.llm_base, trust_remote_code=True, use_fast=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            self.llm_base,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            quantization_config=quantization_config if torch.cuda.is_available() else None,
        )
        self.llm_model = PeftModel.from_pretrained(model, str(self.adapter_path))
        self.llm_model.eval()

    def encoder_predict(self, text: str) -> tuple[bool, float]:
        self.load_encoder()
        assert self.encoder_tokenizer is not None and self.encoder_model is not None
        inputs = self.encoder_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            inputs = {key: value.cuda() for key, value in inputs.items()}
        with torch.no_grad():
            logits = self.encoder_model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0]
        confidence = float(probs[1].detach().cpu())
        return confidence >= self.encoder_threshold, confidence

    def chat_prompt(self, user_prompt: str) -> str:
        assert self.llm_tokenizer is not None
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt}]
        try:
            return self.llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            return f"{SYSTEM_PROMPT}\n\n用户：{user_prompt}\n\n助手："

    def llm_predict(self, text: str, candidate_entities: list[str]) -> tuple[dict[str, Any], bool, str]:
        self.load_llm()
        assert self.llm_tokenizer is not None and self.llm_model is not None
        candidate_hint = "候选实体：" + "；".join(candidate_entities) if candidate_entities else "候选实体：无"
        user_prompt = build_user_prompt(f"{text}\n{candidate_hint}")
        prompt = self.chat_prompt(user_prompt)
        device = next(self.llm_model.parameters()).device
        inputs = self.llm_tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            generated = self.llm_model.generate(
                **inputs,
                max_new_tokens=160,
                do_sample=False,
                pad_token_id=self.llm_tokenizer.pad_token_id,
                eos_token_id=self.llm_tokenizer.eos_token_id,
            )
        new_tokens = generated[0, inputs.input_ids.shape[-1] :]
        raw_output = self.llm_tokenizer.decode(new_tokens, skip_special_tokens=True)
        parsed, invalid = parse_model_output(raw_output)
        return parsed, invalid, raw_output

    def score(self, text: str, *, text_id: str | None = None, force_llm: bool = False) -> dict[str, Any]:
        start = time.perf_counter()
        request_id = str(uuid.uuid4())
        text_id = text_id or request_id
        encoder_positive, encoder_confidence = self.encoder_predict(text)
        candidate_entities = extract_candidate_entities(text)
        stage = "encoder_pass"
        parsed = {"has_negative": False, "entities": []}
        invalid_json = False
        raw_output = ""
        if force_llm or encoder_positive:
            stage = "llm_review"
            parsed, invalid_json, raw_output = self.llm_predict(text, candidate_entities)
        parsed_entities = parsed.get("entities", [])
        pred_entities = filter_entities_by_candidates(parsed_entities, text, candidate_entities)
        parsed_entity_keys = {normalize_entity(entity) for entity in parsed_entities}
        pred_entity_keys = {normalize_entity(entity) for entity in pred_entities}
        hallucinated_entity = bool(parsed_entity_keys - pred_entity_keys)
        model_has_negative = bool(parsed.get("has_negative", False)) if stage == "llm_review" else encoder_positive
        confidence = max(encoder_confidence, 0.85 if pred_entities else encoder_confidence)
        risk_events = build_risk_events(text, pred_entities, confidence=confidence, candidate_entities=candidate_entities)
        entity_missing_review = bool(model_has_negative and not risk_events)
        latency = time.perf_counter() - start
        result = {
            "request_id": request_id,
            "text_id": text_id,
            "has_negative": bool(model_has_negative or risk_events),
            "model_has_negative": bool(model_has_negative),
            "entity_missing_review": entity_missing_review,
            "hallucinated_entity": hallucinated_entity,
            "stage": stage,
            "encoder_confidence": round(encoder_confidence, 4),
            "risk_events": risk_events,
            "latency_sec": latency,
            "raw": {
                "llm_output": raw_output,
                "candidate_entities": candidate_entities,
                "parsed_entities": parsed_entities,
                "filtered_entities": pred_entities,
                "invalid_json": invalid_json,
            },
        }
        for event in risk_events:
            self.db.insert_risk_event(text_id, event)
        if entity_missing_review:
            self.db.enqueue_review_case(
                text_id=text_id,
                reason="entity_missing_or_filtered",
                payload=result,
            )
        self.db.refresh_entity_profiles()
        self.db.log_prediction(request_id, text, stage, latency, result, invalid_json, hallucinated_entity)
        return result

    def entity_profile(self, entity: str) -> dict[str, Any] | None:
        return self.db.get_entity_profile(entity)

    def review_queue(self, status: str = "pending", limit: int = 50) -> list[dict[str, Any]]:
        return self.db.review_queue(status=status, limit=limit)

    def metrics(self) -> dict[str, Any]:
        return self.db.metrics()


service = RiskWarningService()
