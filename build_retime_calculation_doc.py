from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


OUTPUT = r"C:\Users\34927\Desktop\acce_data\机械臂轨迹加速实际计算流程.docx"


def set_cell_shading(cell, fill):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = tc_pr.find(qn("w:shd"))
    if shd is None:
        shd = OxmlElement("w:shd")
        tc_pr.append(shd)
    shd.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120):
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{name}"))
        if node is None:
            node = OxmlElement(f"w:{name}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def set_table_borders(table, color="D9D9D9", size="6"):
    tbl_pr = table._tbl.tblPr
    borders = tbl_pr.first_child_found_in("w:tblBorders")
    if borders is None:
        borders = OxmlElement("w:tblBorders")
        tbl_pr.append(borders)
    for edge in ("top", "left", "bottom", "right", "insideH", "insideV"):
        tag = borders.find(qn(f"w:{edge}"))
        if tag is None:
            tag = OxmlElement(f"w:{edge}")
            borders.append(tag)
        tag.set(qn("w:val"), "single")
        tag.set(qn("w:sz"), size)
        tag.set(qn("w:color"), color)


def set_repeat_table_header(row):
    tr_pr = row._tr.get_or_add_trPr()
    tbl_header = OxmlElement("w:tblHeader")
    tbl_header.set(qn("w:val"), "true")
    tr_pr.append(tbl_header)


def style_run(run, font_name="Microsoft YaHei", size=10.5, bold=False, color="000000"):
    run.font.name = font_name
    run._element.get_or_add_rPr().rFonts.set(qn("w:eastAsia"), font_name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:ascii"), font_name)
    run._element.get_or_add_rPr().rFonts.set(qn("w:hAnsi"), font_name)
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor.from_string(color)


def add_body(doc, text, bold_lead=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.25
    if bold_lead and text.startswith(bold_lead):
        r1 = p.add_run(bold_lead)
        style_run(r1, bold=True)
        r2 = p.add_run(text[len(bold_lead):])
        style_run(r2)
    else:
        r = p.add_run(text)
        style_run(r)
    return p


def add_equation(doc, text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(7)
    r = p.add_run(text)
    style_run(r, font_name="Cambria Math", size=11.5)
    return p


def add_bullets(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Bullet")
        p.paragraph_format.space_after = Pt(3)
        p.paragraph_format.line_spacing = 1.2
        style_run(p.add_run(item))


def add_numbered(doc, items):
    for item in items:
        p = doc.add_paragraph(style="List Number")
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.line_spacing = 1.2
        style_run(p.add_run(item))


def add_table(doc, headers, rows, widths=None):
    table = doc.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    set_table_borders(table)
    header = table.rows[0]
    set_repeat_table_header(header)
    for i, text in enumerate(headers):
        cell = header.cells[i]
        set_cell_shading(cell, "1F4E78")
        set_cell_margins(cell)
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        p = cell.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(0)
        style_run(p.add_run(text), size=9.5, bold=True, color="FFFFFF")
        if widths:
            cell.width = Cm(widths[i])
    for ridx, row in enumerate(rows):
        cells = table.add_row().cells
        for i, text in enumerate(row):
            cell = cells[i]
            set_cell_margins(cell)
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            if ridx % 2:
                set_cell_shading(cell, "F2F6FA")
            p = cell.paragraphs[0]
            p.paragraph_format.space_after = Pt(0)
            p.paragraph_format.line_spacing = 1.15
            if i == 0 and len(headers) > 2:
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            style_run(p.add_run(str(text)), size=9.2)
            if widths:
                cell.width = Cm(widths[i])
    after = doc.add_paragraph()
    after.paragraph_format.space_after = Pt(2)
    return table


doc = Document()
section = doc.sections[0]
section.top_margin = Cm(1.8)
section.bottom_margin = Cm(1.8)
section.left_margin = Cm(2.1)
section.right_margin = Cm(2.1)

styles = doc.styles
normal = styles["Normal"]
normal.font.name = "Microsoft YaHei"
normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
normal.font.size = Pt(10.5)

title_style = styles["Title"]
title_style.font.name = "Microsoft YaHei"
title_style._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
title_style.font.size = Pt(22)
title_style.font.bold = True
title_style.font.color.rgb = RGBColor(0, 0, 0)

for name, size in (("Heading 1", 15), ("Heading 2", 12)):
    st = styles[name]
    st.font.name = "Microsoft YaHei"
    st._element.rPr.rFonts.set(qn("w:eastAsia"), "Microsoft YaHei")
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor(0, 0, 0)
    st.paragraph_format.space_before = Pt(12 if name == "Heading 1" else 8)
    st.paragraph_format.space_after = Pt(6)
    st.paragraph_format.keep_with_next = True

p = doc.add_paragraph(style="Title")
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(10)
p.add_run("机械臂轨迹加速实际计算流程")

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
p.paragraph_format.space_after = Pt(16)
r = p.add_run("样条路径  速度规划  时间映射  重新采样")
style_run(r, size=10.5, color="555555")

add_body(doc, "这套算法分成两个独立阶段。样条先根据全部原始关节角建立连续路径 Q(s)，随后速度规划器在这条固定路径上计算每个位置允许的路径速度和路径加速度。最后，程序按新的 30 Hz 时间轴重新取样，生成更少的关节角点。样条决定走哪条路径，速度规划决定多快走完。")

doc.add_heading("核心结论", level=1)
add_table(doc,
          ["对象", "回答的问题", "是否改变路径"],
          [
              ["样条 Q(s)", "原轨迹位置 s 对应什么关节角", "确定并固定连续关节路径"],
              ["路径速度 ṡ", "每个新控制周期沿原轨迹推进多少", "不改变路径，只改变推进速度"],
              ["路径加速度 s̈", "局部倍速增加或降低得多快", "不改变路径，只控制加减速"],
              ["重新采样", "新时间 τ 下输出哪个关节角", "从 Q(s) 读取新的采样点"],
          ], widths=[3.2, 7.3, 5.3])

doc.add_heading("变量定义", level=1)
add_table(doc,
          ["符号", "物理含义", "当前代码中的单位"],
          [
              ["s", "原始轨迹时间，同时作为路径进度坐标", "秒"],
              ["τ", "加速后的真实执行时间", "秒"],
              ["Q(s)", "原轨迹位置 s 对应的六个关节角", "弧度，说明中常换算为度"],
              ["ṡ = ds/dτ", "局部加速倍数", "无量纲"],
              ["s̈ = d²s/dτ²", "局部加速倍数的变化率", "1/秒"],
              ["Q′(s)", "关节角相对原轨迹时间的变化率", "弧度/秒"],
              ["Q″(s)", "上述变化率相对原轨迹时间的变化率", "弧度/秒²"],
          ], widths=[3.2, 8.0, 4.6])

doc.add_heading("实际计算顺序", level=1)

doc.add_heading("读取原始关节角", level=2)
add_body(doc, "输入数据以固定帧率记录。30 Hz 数据中，第 i 帧的原轨迹时间为 sᵢ = i / 30。每个采样点包含六个关节角 qᵢ。")
add_equation(doc, "原始数据  (s₀, q₀), (s₁, q₁), …, (sₙ, qₙ)")

doc.add_heading("一次性拟合连续路径", level=2)
add_body(doc, "程序使用全部原始关节角拟合六个连续样条函数，组合成六维关节路径 Q(s)。样条完成后不再重新拟合。它允许程序查询原采样点之间任意 s 对应的关节角，并计算 Q′(s) 和 Q″(s)。")
add_equation(doc, "Q(s) = [Q₁(s), Q₂(s), Q₃(s), Q₄(s), Q₅(s), Q₆(s)]")

doc.add_heading("建立固定规划网格", level=2)
add_body(doc, "程序在原轨迹时间轴上建立一组固定的规划位置 s₀, s₁, …。默认每两个原始帧之间再细分 4 份。这些位置在速度规划开始前已经全部确定，它们不是最终的 30 Hz 输出点。")
add_equation(doc, "原始帧  0.000 ───────── 0.033    规划网格  0.000 → 0.008 → 0.017 → 0.025 → 0.033")

doc.add_heading("计算每个网格点的运动约束", level=2)
add_body(doc, "在每个固定 sₖ 上，程序计算 Q′(sₖ) 和 Q″(sₖ)。如果路径按 ṡ 推进，并以 s̈ 加速，则六个关节的实际速度和加速度为：")
add_equation(doc, "q̇ = Q′(s) ṡ")
add_equation(doc, "q̈ = Q″(s) ṡ² + Q′(s) s̈")
add_body(doc, "每个关节必须同时满足速度和加速度上限。margin 是规划安全系数，默认 0.98；frame_step 是局部倍速上限，默认 3.0。")
add_equation(doc, "|q̇ⱼ| ≤ Vmax,ⱼ × margin      |q̈ⱼ| ≤ Amax,ⱼ × margin      ṡ ≤ frame_step")

doc.add_heading("反向计算未来是否能够减速", level=2)
add_body(doc, "程序从加速段终点向起点反推。它检查如果前方存在较低速度上限、1 倍速吸盘保护区或终点要求，当前位置最多允许多大的 ṡ，才能在关节最大减速度内及时降速。反向过程得到每个规划位置的可达速度范围。")

doc.add_heading("正向选择最快可行速度", level=2)
add_body(doc, "程序再从起点向终点推进。在反向计算得到的范围内，每一步尽量选择最大的可行 s̈，并更新下一规划位置的速度。代码使用 x = ṡ² 和 u = s̈：")
add_equation(doc, "xₖ₊₁ = xₖ + 2 uₖ (sₖ₊₁ - sₖ)")
add_body(doc, "完成正向计算后，整条加速段的 ṡ 和 s̈ 已经全部确定。实际执行时不再逐点重新规划。")

doc.add_heading("得到新时间与原轨迹位置的映射", level=2)
add_body(doc, "已知相邻规划位置和两端速度后，可以计算走完该小段需要的新执行时间。所有小段时间累加后，得到完整映射 s(τ)。")
add_equation(doc, "Δτₖ = 2 (sₖ₊₁ - sₖ) / (ṡₖ + ṡₖ₊₁)")

doc.add_heading("按新时间轴重新采样", level=2)
add_body(doc, "输出仍为 30 Hz，因此新时间为 τₙ = n / 30。程序先从 s(τ) 查询当前新时间对应的原轨迹位置，再从样条读取关节角。")
add_equation(doc, "sₙ = s(τₙ)      qnew,n = Q(sₙ)")
add_body(doc, "如果原轨迹为 10 秒、优化后为 6 秒，原来约有 300 个采样点，输出约有 180 个采样点。普通加速区的关节角由样条重新计算，吸盘保护区直接保留原始数据行。")

doc.add_heading("完整计算示例", level=1)
add_body(doc, "以下示例只使用两个关节，演示单个规划位置的计算。真实代码同时计算六个关节，并取最严格的公共限制。")
add_table(doc,
          ["参数", "J1", "J2"],
          [
              ["Q′(s)", "20 deg/s", "10 deg/s"],
              ["Q″(s)", "4 deg/s²", "-2 deg/s²"],
              ["最大角速度", "30 deg/s", "20 deg/s"],
              ["最大角加速度", "60 deg/s²", "40 deg/s²"],
          ], widths=[5.0, 5.4, 5.4])

add_body(doc, "速度限制先给出局部 ṡ 上限。J1 允许 30 / 20 = 1.5 倍，J2 允许 20 / 10 = 2.0 倍；若 frame_step = 3，则当前位置的速度上限为 1.5 倍。")
add_equation(doc, "ṡmax = min(1.5, 2.0, 3.0) = 1.5")

add_body(doc, "假设当前位置 ṡ = 1，因此 x = ṡ² = 1。把关节加速度限制代入 q̈ = Q″x + Q′u，可得到：")
add_equation(doc, "J1  -60 ≤ 4 + 20u ≤ 60      得到 -3.2 ≤ u ≤ 2.8")
add_equation(doc, "J2  -40 ≤ -2 + 10u ≤ 40     得到 -3.8 ≤ u ≤ 4.2")
add_body(doc, "两个关节共同允许的路径加速度范围是 -3.2 ≤ s̈ ≤ 2.8。若反向检查确认前方有足够距离，正向过程可以选择 s̈ = 2.8。假设规划网格间隔 Δs = 0.05：")
add_equation(doc, "ṡnext² = 1² + 2 × 2.8 × 0.05 = 1.28")
add_equation(doc, "ṡnext = √1.28 = 1.131")
add_body(doc, "这表示路径速度由 1.000 倍增加到约 1.131 倍。若前方需要降至 1 倍速，反向过程会降低当前位置允许的速度，正向过程也会选择较小或为负的 s̈。")

doc.add_heading("抽帧和样条重定时的区别", level=1)
add_table(doc,
          ["项目", "直接抽帧", "样条重定时"],
          [
              ["关节角来源", "只选择原始采样点", "从连续 Q(s) 计算任意位置"],
              ["倍速", "通常使用整数或跳变的帧步长", "可以连续变化，例如 1.0 到 1.7 到 2.3"],
              ["速度规划", "帧步长隐含局部倍速", "显式计算 ṡ"],
              ["加减速规划", "难以平滑控制", "显式计算 s̈ 并提前减速"],
              ["约束", "抽帧后再检查是否越限", "规划时同时满足速度和加速度限制"],
          ], widths=[3.3, 6.2, 6.3])

doc.add_heading("容易混淆的概念", level=1)
add_bullets(doc, [
    "样条只拟合一次，用于确定连续关节路径 Q(s)。",
    "规划网格中的 sₖ 是预先建立的原轨迹位置，不是最终输出采样点。",
    "反向和正向过程在固定规划网格上计算整条 ṡ 和 s̈ 曲线。",
    "最终输出点由固定的新时间 τₙ = n / 30 决定，再通过 s(τₙ) 和 Q(sₙ) 计算关节角。",
    "frame_step = 3 表示局部最多 3 倍，不保证整条轨迹达到 3 倍。关节速度、加速度和保护区通常会降低实际倍速。",
    "当前代码使用原始时间作为 s。它是原轨迹时间坐标，在优化中承担路径进度参数的作用。",
])

doc.add_heading("一句话流程", level=1)
add_body(doc, "先用全部原始关节角拟合连续路径 Q(s)，再在固定 s 网格上根据六轴速度和加速度限制反向、正向求出整条 ṡ 与 s̈，随后得到 s(τ)，最后按新的 30 Hz 时间轴输出 Q(s(τ))。")

doc.core_properties.title = "机械臂轨迹加速实际计算流程"
doc.core_properties.subject = "轨迹重定时算法说明"
doc.core_properties.keywords = "机械臂, 轨迹加速, 样条, TOPP, 重定时"
doc.save(OUTPUT)
print(OUTPUT)
