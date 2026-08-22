#!/usr/bin/env python3
"""Narrow decoding safeguards for literal CHURRO transcription."""

from __future__ import annotations

import torch
from transformers import LogitsProcessor, LogitsProcessorList, RepetitionPenaltyLogitsProcessor


class TargetedOCRLoopGuard(LogitsProcessor):
    """Break only clear generation loops instead of globally banning n-grams.

    The guard blocks one token when the response is about to continue either a
    long single-token run or a complete multi-token block already repeated
    three times consecutively.  Ordinary repeated words and recurring XML line
    tags do not match the complete-block condition and are left untouched.
    """

    def __init__(
        self,
        prompt_length: int,
        minimum_period: int = 4,
        maximum_period: int = 96,
        required_repeats: int = 3,
        single_token_run: int = 12,
    ) -> None:
        if prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")
        if minimum_period < 2 or maximum_period < minimum_period:
            raise ValueError("invalid loop period bounds")
        if required_repeats < 3 or single_token_run < 4:
            raise ValueError("loop thresholds are too permissive")
        self.prompt_length = prompt_length
        self.minimum_period = minimum_period
        self.maximum_period = maximum_period
        self.required_repeats = required_repeats
        self.single_token_run = single_token_run
        self.events = 0

    def _banned_token(self, sequence: torch.Tensor) -> int | None:
        generated = sequence[self.prompt_length :]
        count = int(generated.numel())
        if count >= self.single_token_run:
            tail = generated[-self.single_token_run :]
            if bool(torch.all(tail == tail[0])):
                return int(tail[0].item())

        maximum = min(self.maximum_period, count // self.required_repeats)
        for period in range(self.minimum_period, maximum + 1):
            span = period * self.required_repeats
            tail = generated[-span:]
            block = tail[:period]
            if all(
                bool(torch.equal(block, tail[index * period : (index + 1) * period]))
                for index in range(1, self.required_repeats)
            ):
                # The sequence currently ends exactly at the repeated block's
                # boundary.  Block only the token that would start it again.
                return int(block[0].item())
        return None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for batch_index in range(input_ids.shape[0]):
            banned = self._banned_token(input_ids[batch_index])
            if banned is not None:
                scores[batch_index, banned] = -torch.inf
                self.events += 1
        return scores


def faithful_logits_processors(prompt_length: int) -> LogitsProcessorList:
    """Use a tiny response-only repetition penalty.

    The evaluator supplies a loose built-in 32-gram guard separately.  It is
    substantially cheaper than inspecting GPU token history in Python at every
    decode step, while still interrupting long exact XML loops.
    """

    return LogitsProcessorList(
        [
            RepetitionPenaltyLogitsProcessor(
                penalty=1.01,
                prompt_ignore_length=prompt_length,
            ),
        ]
    )
