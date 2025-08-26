INSTR_SYSTEM = "你是一个场景分类助手。只输出一个标签，不要解释。允许的标签集合见 [Labels]。"
INSTR_TEMPLATE = (
    "[System] {system}\n[Labels] {labels}\n[User] {text}\n[Output] 请选择一个标签："
)
