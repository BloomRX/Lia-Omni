"""Emotion-conditioned LoRA trainer for pinned Qwen3-TTS Base.

Runtime entry point: run this file with the same CLI arguments as the patched
community ``sft_12hz_lora.py`` trainer. It injects the same user-role emotion
instruction embeddings that Qwen's generation path prepends, and keeps the
standard codec/SFT losses. The adapter can then be tested through Base ICL
inference; do not force the model into ``custom_voice`` mode.
"""
from __future__ import annotations

import collections
import sys
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

# The notebook puts the pinned Qwen finetuning directory on PYTHONPATH.
import sft_12hz_lora as trainer

EMOTION_INSTRUCTIONS = {
    "alegre": "Fale em português brasileiro com alegria calorosa e genuína, energia expressiva e sorriso na voz, sem gritar.",
    "comemorativa": "Fale em português brasileiro em tom de comemoração, com entusiasmo evidente e energia alta, sem gritar.",
    "neutro": "Fale em português brasileiro de forma neutra, natural e conversacional, sem dramatizar.",
    "calma": "Fale em português brasileiro com calma, serenidade e ritmo tranquilo, mantendo naturalidade.",
    "assertiva": "Fale em português brasileiro com firmeza e confiança, de forma clara e assertiva, sem soar agressiva.",
    "triste": "Fale em português brasileiro com tristeza contida e emoção sincera, sem exagerar nem sussurrar.",
    "irritada": "Fale em português brasileiro com irritação perceptível e firmeza, sem gritar.",
}


def _emotion_for(row: dict[str, Any]) -> str:
    key = str(row.get("emotion_audit", "")).strip().lower()
    if key not in EMOTION_INSTRUCTIONS:
        raise ValueError(f"Exemplo sem emoção condicionável/revisada: {key!r}")
    return key


class EmotionConditionedDataset(trainer.TTSDataset):
    """Adds user-role emotion tokens to each reviewed audio/text training pair."""

    def __getitem__(self, idx):
        item = self.data_list[idx]
        label = _emotion_for(item)
        base = super().__getitem__(idx)
        prompt = f"<|im_start|>user\n{EMOTION_INSTRUCTIONS[label]}<|im_end|>\n"
        encoded = self.processor(text=prompt, return_tensors="pt", padding=True)
        base["emotion_ids"] = encoded["input_ids"][0].to(dtype=torch.long)
        base["emotion_label"] = label
        return base

    def collate_fn(self, batch):
        result = super().collate_fn(batch)
        max_len = max(item["emotion_ids"].numel() for item in batch)
        batch_size = len(batch)
        ids = torch.zeros((batch_size, max_len), dtype=torch.long)
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool)
        for i, item in enumerate(batch):
            n = item["emotion_ids"].numel()
            ids[i, :n] = item["emotion_ids"]
            mask[i, :n] = True
        result["emotion_ids"] = ids
        result["emotion_mask"] = mask
        return result


def compute_emotion_conditioned_loss(model, batch):
    """Teacher-force speech while prepending the same instruction embedding used by generate()."""
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]
    emotion_ids = batch["emotion_ids"]
    emotion_mask = batch["emotion_mask"]

    with torch.no_grad():
        speaker_embedding = model.speaker_encoder(
            ref_mels.to(dtype=next(model.parameters()).dtype, device=next(model.parameters()).device)
        ).detach()
    if model.training and trainer.target_speaker_embedding is None:
        trainer.target_speaker_embedding = speaker_embedding.detach().to("cpu")

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]
    input_text_embedding = model.talker.model.text_embedding(input_text_ids)
    if hasattr(model.talker, "text_projection"):
        input_text_embedding = model.talker.text_projection(input_text_embedding)
    input_text_embedding = input_text_embedding * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    input_codec_embedding[:, 6, :] = speaker_embedding
    input_embeddings = input_text_embedding + input_codec_embedding
    for i in range(1, 16):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        input_embeddings = input_embeddings + codec_i_embedding * codec_mask.unsqueeze(-1)

    # This is the same path as Qwen3-TTS Base generation:
    # text_projection(text_embedding(instruct_ids)) is prepended before speech inputs.
    emotion_embeddings = model.talker.text_projection(
        model.talker.get_text_embeddings()(emotion_ids)
    )
    emotion_embeddings = emotion_embeddings * emotion_mask.unsqueeze(-1)
    batch_size, emotion_len = emotion_ids.shape
    input_embeddings = torch.cat([emotion_embeddings, input_embeddings], dim=1)
    attention_mask = torch.cat([emotion_mask.to(attention_mask.dtype), attention_mask], dim=1)
    codec_0_labels = torch.cat(
        [codec_0_labels.new_full((batch_size, emotion_len), -100), codec_0_labels], dim=1
    )
    codec_ids = torch.cat(
        [codec_ids.new_zeros((batch_size, emotion_len, codec_ids.shape[-1])), codec_ids], dim=1
    )
    codec_mask = torch.cat(
        [codec_mask.new_zeros((batch_size, emotion_len)), codec_mask], dim=1
    )

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=None,
        output_hidden_states=True,
    )
    logits = outputs.logits
    targets = codec_0_labels[:, 1:]
    codec_0_loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
    )
    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]
    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
        talker_codec_ids, talker_hidden_states
    )
    return codec_0_loss + sub_talker_loss


def _install_balanced_dataloader():
    original_loader = trainer.DataLoader

    def loader(dataset, *args, **kwargs):
        if isinstance(dataset, EmotionConditionedDataset) and kwargs.get("shuffle", False):
            labels = [_emotion_for(row) for row in dataset.data_list]
            counts = collections.Counter(labels)
            weights = [1.0 / counts[label] for label in labels]
            batch_size = kwargs.pop("batch_size", args[0] if args else 1)
            kwargs.pop("shuffle", None)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return original_loader(dataset, batch_size=batch_size, sampler=sampler, **kwargs)
        return original_loader(dataset, *args, **kwargs)

    trainer.DataLoader = loader


def main():
    trainer.TTSDataset = EmotionConditionedDataset
    trainer.compute_loss = compute_emotion_conditioned_loss
    _install_balanced_dataloader()
    trainer.train()


if __name__ == "__main__":
    main()
