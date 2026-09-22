"""vLLM 0.19.1 reasoning-parser 插件：回移上游 vLLM PR #35687
("[Bugfix] Treat <tool_call> as implicit reasoning end in Qwen3 parser"，v0.20.0 起合入)。

问题：Qwen3.5 开思考时，有时在思考里直接写 <tool_call>、没写 </think> 就结束；0.19.1 的 qwen3 解析器把整段当思考
→ 正文为空、工具调用丢失 → tau2 判定"既无正文也无工具调用"，整段对话重开。实测约 13% 的思考轮出现空回复。
修法（与上游逐行一致）：没有 </think> 时，把第一个 <tool_call> 当作思考的隐式结束，从它开始算正文。
  - 正常写了 </think> 的输出：完全不变（思考里的草稿调用仍算思考）
  - 没写 </think> 也没有 <tool_call>：完全不变（仍整段当思考；这部分由 lean_agent 的"补 </think> 续写"兜住）
  - 提示词里成对的 <tool_call>…</tool_call>（历史轮次 / 模板示例）不算思考结束
效果：空回复 13.3% → 7.0%。
用法：vllm serve ... --reasoning-parser-plugin qwen3_toolend_reasoning_parser.py --reasoning-parser qwen3_toolend
"""
from collections.abc import Iterable, Sequence

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser


@ReasoningParserManager.register_module("qwen3_toolend")
class Qwen3ToolEndReasoningParser(Qwen3ReasoningParser):
    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self._tool_call_tag = "<tool_call>"
        self._tool_call_token_id = self.vocab.get(self._tool_call_tag)
        self._tool_call_end_tag = "</tool_call>"
        self._tool_call_end_token_id = self.vocab.get(self._tool_call_end_tag)

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        tool_call_token_id = self._tool_call_token_id
        tool_call_end_token_id = self._tool_call_end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id == start_token_id:
                return False
            if token_id == end_token_id:
                return True
            if tool_call_token_id is not None and token_id == tool_call_token_id:
                # 成对出现的是提示词里的示例 / 历史，不是模型这次的输出
                if tool_call_end_token_id is not None and any(
                    input_ids[j] == tool_call_end_token_id for j in range(i + 1, len(input_ids))
                ):
                    continue
                return True
        return False

    def is_reasoning_end_streaming(self, input_ids: Sequence[int], delta_ids: Iterable[int]) -> bool:
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True
        if self._tool_call_token_id is not None:
            return self._tool_call_token_id in delta_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        result = super().extract_content_ids(input_ids)
        if result:
            return result
        if self._tool_call_token_id is not None and self._tool_call_token_id in input_ids:
            tool_call_index = len(input_ids) - 1 - input_ids[::-1].index(self._tool_call_token_id)
            return input_ids[tool_call_index:]
        return []

    def extract_reasoning(self, model_output: str, request) -> tuple[str | None, str | None]:
        model_output_parts = model_output.partition(self.start_token)
        model_output = model_output_parts[2] if model_output_parts[1] else model_output_parts[0]

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            return reasoning, content or None

        if not self.thinking_enabled:
            return None, model_output

        # 没有 </think>：看有没有 <tool_call> 作为隐式结束
        tool_call_index = model_output.find(self._tool_call_tag)
        if tool_call_index != -1:
            reasoning = model_output[:tool_call_index]
            content = model_output[tool_call_index:]
            return reasoning or None, content or None
        return model_output, None

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if self.start_token_id in delta_token_ids:
            start_idx = delta_text.find(self.start_token)
            if start_idx >= 0:
                delta_text = delta_text[start_idx + len(self.start_token):]

        if self.end_token_id in delta_token_ids:
            end_index = delta_text.find(self.end_token)
            if end_index >= 0:
                reasoning = delta_text[:end_index]
                content = delta_text[end_index + len(self.end_token):]
                if not reasoning and not content:
                    return None
                return DeltaMessage(reasoning=reasoning if reasoning else None, content=content if content else None)
            return None

        if self._tool_call_token_id is not None and self._tool_call_token_id in delta_token_ids:
            tool_index = delta_text.find(self._tool_call_tag)
            if tool_index >= 0:
                reasoning = delta_text[:tool_index]
                content = delta_text[tool_index:]
                return DeltaMessage(reasoning=reasoning if reasoning else None, content=content if content else None)

        if not delta_text:
            return None
        elif self.end_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)
        elif self._tool_call_token_id is not None and self._tool_call_token_id in previous_token_ids:
            return DeltaMessage(content=delta_text)
        else:
            return DeltaMessage(reasoning=delta_text)
