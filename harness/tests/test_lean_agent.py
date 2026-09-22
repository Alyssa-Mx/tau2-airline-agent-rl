"""离线单测：把 generate 桩掉，不需要 GPU / 网络 / vLLM。需要安装 tau2（τ²-bench）。
验证：工具返回紧凑格式、旧返回占位与按实体钉住、刷屏占位、写库工具存根 → 展开重生成、补 </think> 续写、不污染共享参数。
    python harness/tests/test_lean_agent.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tau2.data_model.message import AssistantMessage, ToolCall, ToolMessage, UserMessage  # noqa: E402

import lean_agent  # noqa: E402

CALLS = []


def fake_generate(model, messages, tools=None, call_name=None, **kw):
    eb = kw.get("extra_body") or {}
    ctk = eb.get("chat_template_kwargs") or {}
    exp = ctk.get("expanded_tools", [])
    CALLS.append({"expanded": list(exp), "ctk": ctk, "extra_body": eb, "messages": messages})
    last = messages[-1]
    txt = getattr(last, "content", "") or ""
    if isinstance(last, AssistantMessage) and txt.startswith("<think>"):        # 续写请求
        return AssistantMessage(role="assistant", content="Sure, could you share your reservation id?")
    if isinstance(last, UserMessage) and "think only" in txt:                  # 思考没收尾、正文为空
        return AssistantMessage(role="assistant", content=None,
                                raw_data={"choices": [{"message": {"reasoning_content": "I should ask for the id."}}]})
    if isinstance(last, UserMessage) and "cancel it" in txt:
        if "cancel_reservation" not in exp:                                     # 模型调了存根工具
            return AssistantMessage(role="assistant", content=None, tool_calls=[ToolCall(id="x1", name="cancel_reservation", arguments={})])
        return AssistantMessage(role="assistant", content=None,
                                tool_calls=[ToolCall(id="x2", name="cancel_reservation", arguments={"reservation_id": "ABC123"})])
    if isinstance(last, UserMessage) and "look up" in txt:
        rid = txt.split()[-1]
        return AssistantMessage(role="assistant", content=None,
                                tool_calls=[ToolCall(id="r" + rid, name="get_reservation_details", arguments={"reservation_id": rid})])
    return AssistantMessage(role="assistant", content="ok")


lean_agent.generate = fake_generate


class FakeTool:
    def __init__(self, n):
        self.name = n
        self.openai_schema = {"type": "function", "function": {"name": n, "description": n, "parameters": {}}}


tools = [FakeTool(n) for n in ["get_reservation_details", "get_user_details", "search_direct_flight",
                               "cancel_reservation", "book_reservation", "calculate"]]
os.environ["LEAN_LOG_DIR"] = tempfile.mkdtemp()
base_args = {"api_base": "http://x/v1", "temperature": 0.7, "extra_body": {"chat_template_kwargs": {"enable_thinking": True}}}
fails = []


def check(name, cond):
    print(f"  {'OK  ' if cond else 'FAIL'} {name}")
    if not cond:
        fails.append(name)


# ---- 工具返回紧凑格式（无损）----
flights = [{"flight_number": "HAT033", "origin": "JFK", "destination": "DTW", "status": "available",
            "scheduled_departure_time_est": "00:00:00", "scheduled_arrival_time_est": "02:00:00+1", "date": None,
            "available_seats": {"basic_economy": 5, "economy": 5, "business": 4},
            "prices": {"basic_economy": 88, "economy": 135, "business": 363}}]
check("航班搜索压成一行", lean_agent.compact_result("search_direct_flight", json.dumps(flights))
      == "HAT033 JFK>DTW 00:00-02:00+1 basic_economy:5@$88 economy:5@$135 business:4@$363")
check("去掉引号和空字段", lean_agent.compact_result(
    "get_reservation_details", json.dumps({"reservation_id": "ABC123", "status": None, "insurance": "no", "nonfree_baggages": 0}))
      == "{reservation_id: ABC123, insurance: no, nonfree_baggages: 0}")
check("非 JSON 原样返回", lean_agent.compact_result("calculate", "Error: bad") == "Error: bad")

# ---- 旧返回占位：保留最近 2 条 + 每个预订号最近一次 ----
agent = lean_agent.create_lean_agent(tools, "POLICY", llm="openai/basemodel", llm_args=base_args)
st = agent.get_init_state()
for rid in ["ABC123", "XYZ999", "ABC123", "QQQ111"]:
    m, st = agent.generate_next_message(UserMessage(role="user", content=f"please look up {rid}"), st)
    tc = m.tool_calls[0]
    _, st = agent.generate_next_message(ToolMessage(id=tc.id, role="tool", content=json.dumps({"reservation_id": rid, "cabin": "economy", "status": None})), st)
check("读工具从不展开写工具", all("cancel_reservation" not in c["expanded"] for c in CALLS))
state_tools = [x for x in st.messages if isinstance(x, ToolMessage)]
check("状态里存的是压缩后的返回", state_tools[0].content == "{reservation_id: ABC123, cabin: economy}")
vt = [x.content for x in agent._view(st) if isinstance(x, ToolMessage)]
check("更早的同实体返回换成占位", vt[0].startswith("[older get_reservation_details"))
check("每个实体最近一次 + 最近 2 条保留", "XYZ999" in vt[1] and "ABC123" in vt[2] and "QQQ111" in vt[3])

# ---- 刷屏占位 + 写库工具存根 → 展开后重生成 ----
st.messages.append(UserMessage(role="user", content="hello?"))
st.messages.append(AssistantMessage(role="assistant", content="!" * 300))
m, st = agent.generate_next_message(UserMessage(role="user", content="yes, cancel it"), st)
check("第一次生成时写工具是存根", "cancel_reservation" not in CALLS[-2]["expanded"])
check("调了存根 → 展开后重生成", "cancel_reservation" in CALLS[-1]["expanded"])
check("最终采用重生成那次的调用", m.tool_calls[0].arguments == {"reservation_id": "ABC123"})
sent = [getattr(x, "content", "") or "" for x in CALLS[-1]["messages"]]
check("发给模型的历史里刷屏换成占位", any("garbled" in s for s in sent) and not any(s.startswith("!!!!") for s in sent))
check("状态里保留原始刷屏（轨迹不改）", any(isinstance(x, AssistantMessage) and (x.content or "").startswith("!!!!") for x in st.messages))
check("思考开关原样透传", CALLS[-1]["ctk"].get("enable_thinking") is True)
check("不污染共享的 llm_args", "expanded_tools" not in agent.llm_args["extra_body"]["chat_template_kwargs"])

# ---- 航班搜索按"航线 + 日期"钉住 ----
s4 = agent.get_init_state()
s4.messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[
    ToolCall(id="s1", name="search_direct_flight", arguments={"origin": "JFK", "destination": "SFO", "date": "2024-05-26"})]))
s4.messages.append(agent._ingest_tool(ToolMessage(id="s1", role="tool", content=json.dumps(flights)), s4))
for i, rid in enumerate(["AAA111", "BBB222", "CCC333"]):
    s4.messages.append(AssistantMessage(role="assistant", content=None, tool_calls=[
        ToolCall(id=f"g{i}", name="get_reservation_details", arguments={"reservation_id": rid})]))
    s4.messages.append(agent._ingest_tool(ToolMessage(id=f"g{i}", role="tool", content=json.dumps({"reservation_id": rid})), s4))
v4 = [x.content for x in agent._view(s4) if isinstance(x, ToolMessage)]
check("三条新返回之后，航班搜索结果仍保留", "HAT033" in v4[0])

# ---- 思考没收尾、正文为空 → 补 </think> 续写 ----
os.environ["LEAN_THINK_CLOSE_CONTINUE"] = "1"
a5 = lean_agent.create_lean_agent(tools, "POLICY", llm="openai/basemodel", llm_args=base_args)
m, _ = a5.generate_next_message(UserMessage(role="user", content="think only"), a5.get_init_state())
check("续写拿到正文", (m.content or "").startswith("Sure"))
cont = CALLS[-1]
check("续写请求：continue_final_message、关思考", cont["extra_body"].get("continue_final_message") is True
      and cont["ctk"].get("enable_thinking") is False)
check("续写的前缀是它自己的思考 + </think>", cont["messages"][-1].content.endswith("</think>\n\n"))

# ---- 全关 = 原版调用 ----
for k in ("LEAN_TOOL_DEFS", "LEAN_COMPACT_RESULTS", "LEAN_MASK_OLDER", "LEAN_DROP_FLOODS", "LEAN_THINK_CLOSE_CONTINUE"):
    os.environ[k] = "0"
a6 = lean_agent.create_lean_agent(tools, "POLICY", llm="openai/basemodel", llm_args=base_args)
a6.generate_next_message(UserMessage(role="user", content="hi"), a6.get_init_state())
check("全关时请求参数与原版相同", CALLS[-1]["extra_body"] == base_args["extra_body"])

print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
