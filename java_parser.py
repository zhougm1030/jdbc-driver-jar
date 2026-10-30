#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于 tree-sitter 的 Java 源码解析器

提取内容：
  - 包名、导入列表
  - 类 / 接口 / 枚举 / 记录 / 注解类型的名称
  - 父类 (extends) 与实现的接口 (implements)
  - 注解（含具名参数与单值参数）
  - 成员变量的类型、名称、修饰符（static / final / public / ...）
  - 方法与构造器签名
  - 嵌套类型、内部类、局部类、匿名类

用法：
    python java_parser.py <file.java | dir> [--json] [--out out.json]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

# --------------------------------------------------------------------------
# 依赖引导：优先使用已安装的包；若缺失，尝试从工程 vendor/ 目录就地加载，
# 保证工程在无外网、无预装依赖的环境下也能直接运行。
# --------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
_VENDOR_LIB = os.path.join(_ROOT, "vendor", "lib")
if os.path.isdir(_VENDOR_LIB) and _VENDOR_LIB not in sys.path:
    sys.path.insert(0, _VENDOR_LIB)

try:
    import tree_sitter_java as tsjava
    from tree_sitter import Language, Node, Parser
except ImportError as exc:  # pragma: no cover
    sys.stderr.write(
        f"[ERROR] 缺少依赖: {exc}\n"
        f"请先执行离线安装:\n"
        f"    python tools/install_deps.py\n"
        f"或联网安装:\n"
        f"    pip install tree-sitter tree-sitter-java\n"
    )
    raise SystemExit(1) from exc

# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------

LANGUAGE = Language(tsjava.language())

#: 修饰符关键字全集
MODIFIER_KEYWORDS = {
    "public", "protected", "private",
    "static", "abstract", "final",
    "native", "synchronized", "transient",
    "volatile", "strictfp", "default",
    "sealed", "non-sealed",
}

#: 声明类型节点 -> 类别名
TYPE_DECL_NODES = {
    "class_declaration": "class",
    "interface_declaration": "interface",
    "enum_declaration": "enum",
    "record_declaration": "record",
    "annotation_type_declaration": "annotation_type",
}


def _text(node: Optional[Node], src: bytes) -> str:
    """取节点源码文本。"""
    if node is None:
        return ""
    return src[node.start_byte:node.end_byte].decode("utf-8", errors="replace")


def _first_child(node: Optional[Node], *types: str) -> Optional[Node]:
    """取第一个类型匹配的子节点。"""
    if node is None:
        return None
    for child in node.children:
        if child.type in types:
            return child
    return None


def _line(node: Node) -> int:
    return node.start_point[0] + 1


# --------------------------------------------------------------------------
# 数据模型
# --------------------------------------------------------------------------

@dataclass
class Annotation:
    """注解，如 @Component(value = "x")"""

    name: str                                   # 注解名（不含 @）
    arguments: Dict[str, str] = field(default_factory=dict)  # 具名参数
    value: Optional[str] = None                 # 单值形式
    line: int = 0

    def __str__(self) -> str:
        if self.arguments:
            args = ", ".join(f"{k} = {v}" for k, v in self.arguments.items())
            return f"@{self.name}({args})"
        if self.value is not None:
            return f"@{self.name}({self.value})"
        return f"@{self.name}"


@dataclass
class Modifiers:
    """修饰符集合 + 注解"""

    keywords: List[str] = field(default_factory=list)
    annotations: List[Annotation] = field(default_factory=list)

    def has(self, *names: str) -> bool:
        return any(n in self.keywords for n in names)

    @property
    def is_static(self) -> bool:
        return "static" in self.keywords

    @property
    def is_final(self) -> bool:
        return "final" in self.keywords

    @property
    def is_abstract(self) -> bool:
        return "abstract" in self.keywords

    @property
    def visibility(self) -> str:
        for v in ("public", "protected", "private"):
            if v in self.keywords:
                return v
        return "package-private"

    def copy(self) -> "Modifiers":
        return Modifiers(keywords=list(self.keywords), annotations=list(self.annotations))

    def __str__(self) -> str:
        return " ".join(self.keywords)


@dataclass
class Field:
    """成员变量"""

    name: str
    type: str                                   # 声明类型文本
    modifiers: Modifiers = field(default_factory=Modifiers)
    value: Optional[str] = None                 # 初始值（无则 None）
    dimensions: int = 0                         # 额外数组维度（int a[] 形式）
    is_enum_constant: bool = False
    implicit_modifiers: List[str] = field(default_factory=list)  # 接口字段隐式 public static final
    line: int = 0

    @property
    def is_static(self) -> bool:
        return self.modifiers.is_static or "static" in self.implicit_modifiers

    @property
    def is_final(self) -> bool:
        return self.modifiers.is_final or "final" in self.implicit_modifiers

    @property
    def is_constant(self) -> bool:
        """常量：static + final（接口字段隐式成立）。"""
        return self.is_static and self.is_final

    @property
    def annotations(self) -> List[Annotation]:
        return self.modifiers.annotations


@dataclass
class Method:
    """方法"""

    name: str
    return_type: str = ""
    parameters: List[str] = field(default_factory=list)
    modifiers: Modifiers = field(default_factory=Modifiers)
    throws: List[str] = field(default_factory=list)
    type_parameters: List[str] = field(default_factory=list)
    default_value: Optional[str] = None
    line: int = 0

    @property
    def annotations(self) -> List[Annotation]:
        return self.modifiers.annotations

    @property
    def signature(self) -> str:
        tp = f"<{', '.join(self.type_parameters)}>" if self.type_parameters else ""
        th = f" throws {', '.join(self.throws)}" if self.throws else ""
        dv = f" default {self.default_value}" if self.default_value else ""
        return f"{tp}{self.return_type} {self.name}({', '.join(self.parameters)}){th}{dv}".strip()


@dataclass
class Constructor:
    """构造器"""

    name: str
    parameters: List[str] = field(default_factory=list)
    modifiers: Modifiers = field(default_factory=Modifiers)
    throws: List[str] = field(default_factory=list)
    line: int = 0

    @property
    def annotations(self) -> List[Annotation]:
        return self.modifiers.annotations

    @property
    def signature(self) -> str:
        th = f" throws {', '.join(self.throws)}" if self.throws else ""
        return f"{self.name}({', '.join(self.parameters)}){th}"


@dataclass
class TypeDecl:
    """类型声明（类/接口/枚举/记录/注解类型/匿名类）"""

    kind: str                                   # class / interface / enum / record / annotation_type / anonymous_class
    name: str
    modifiers: Modifiers = field(default_factory=Modifiers)
    superclass: Optional[str] = None
    interfaces: List[str] = field(default_factory=list)
    extends_interfaces: List[str] = field(default_factory=list)
    fields: List[Field] = field(default_factory=list)
    methods: List[Method] = field(default_factory=list)
    constructors: List[Constructor] = field(default_factory=list)
    enum_constants: List[str] = field(default_factory=list)
    record_components: List[str] = field(default_factory=list)
    nested_types: List["TypeDecl"] = field(default_factory=list)
    line: int = 0
    end_line: int = 0

    @property
    def annotations(self) -> List[Annotation]:
        return self.modifiers.annotations

    @property
    def is_static(self) -> bool:
        return self.modifiers.is_static

    @property
    def is_abstract(self) -> bool:
        return self.modifiers.is_abstract

    @property
    def constants(self) -> List[Field]:
        return [f for f in self.fields if f.is_constant]


@dataclass
class JavaFile:
    """一个 Java 文件的解析结果"""

    path: str = ""
    package: Optional[str] = None
    imports: List[str] = field(default_factory=list)
    types: List[TypeDecl] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def iter_types(self) -> Iterator[TypeDecl]:
        """递归遍历所有类型（含嵌套类型、局部类）。"""
        stack = list(reversed(self.types))
        while stack:
            t = stack.pop()
            yield t
            stack.extend(reversed(t.nested_types))

    def find_type(self, name: str) -> Optional[TypeDecl]:
        for t in self.iter_types():
            if t.name == name:
                return t
        return None


# --------------------------------------------------------------------------
# 解析实现
# --------------------------------------------------------------------------

class JavaParser:
    def __init__(self) -> None:
        self._parser = Parser(LANGUAGE)
        self._src: bytes = b""

    # ---------------- 修饰符 / 注解 ----------------

    def _parse_modifiers(self, node: Optional[Node]) -> Modifiers:
        mods = Modifiers()
        if node is None:
            return mods
        for child in node.children:
            if child.type in ("marker_annotation", "annotation"):
                ann = self._parse_annotation(child)
                if ann:
                    mods.annotations.append(ann)
            elif child.type in MODIFIER_KEYWORDS:
                mods.keywords.append(child.type)
        return mods

    def _parse_annotation(self, node: Node) -> Optional[Annotation]:
        name_node = _first_child(node, "identifier", "scoped_identifier")
        if name_node is None:
            return None
        ann = Annotation(name=_text(name_node, self._src), line=_line(node))

        args_node = _first_child(node, "annotation_argument_list")
        if args_node is not None:
            for child in args_node.children:
                if child.type == "element_value_pair":
                    key_node = _first_child(child, "identifier")
                    val_node = None
                    for c in child.children:
                        if c.type not in ("identifier", "="):
                            val_node = c
                            break
                    if key_node is not None:
                        ann.arguments[_text(key_node, self._src)] = _text(val_node, self._src)
                elif child.type not in ("(", ")", ","):
                    ann.value = _text(child, self._src)
        return ann

    # ---------------- 类型引用 ----------------

    def _parse_type_list(self, node: Optional[Node]) -> List[str]:
        """解析 type_list（implements / extends 列表）。"""
        if node is None:
            return []
        type_list = _first_child(node, "type_list") or node
        return [_text(c, self._src) for c in type_list.children if c.type != ","]

    def _parse_superclass(self, node: Optional[Node]) -> Optional[str]:
        if node is None:
            return None
        for c in node.children:
            if c.type != "extends":
                return _text(c, self._src)
        return None

    # ---------------- 成员变量 ----------------

    def _parse_field(self, node: Node, implicit: Optional[List[str]] = None) -> List[Field]:
        """field_declaration / constant_declaration -> Field 列表（支持 int a = 1, b = 2）。"""
        mods = self._parse_modifiers(_first_child(node, "modifiers"))
        type_node = None
        declarators: List[Node] = []
        for c in node.children:
            if c.type in ("modifiers", ";"):
                continue
            if c.type == "variable_declarator":
                declarators.append(c)
            elif type_node is None:
                type_node = c

        type_text = _text(type_node, self._src) if type_node else ""
        results: List[Field] = []
        for decl in declarators:
            name_node = _first_child(decl, "identifier")
            value_node = None
            dims = 0
            for c in decl.children:
                if c.type == "identifier":
                    continue
                if c.type == "dimensions":
                    dims += 1
                elif c.type == "=":
                    continue
                elif value_node is None:
                    value_node = c
            results.append(Field(
                name=_text(name_node, self._src),
                type=type_text,
                modifiers=mods.copy(),
                value=_text(value_node, self._src) if value_node is not None else None,
                dimensions=dims,
                implicit_modifiers=list(implicit or []),
                line=_line(decl),
            ))
        return results

    # ---------------- 方法 / 构造器 ----------------

    def _parse_method(self, node: Node) -> Method:
        mods = self._parse_modifiers(_first_child(node, "modifiers"))
        name_node = _first_child(node, "identifier")
        return_type = ""
        type_params: List[str] = []
        params: List[str] = []
        throws: List[str] = []
        default_value: Optional[str] = None

        for c in node.children:
            if c.type == "modifiers":
                continue
            if c.type == "type_parameters":
                type_params = [
                    _text(x, self._src) for x in c.children
                    if x.type not in ("<", ">")
                ]
            elif c.type == "formal_parameters":
                params = [self._param_text(p) for p in c.children
                          if p.type in ("formal_parameter", "spread_parameter")]
            elif c.type == "throws":
                throws = [_text(x, self._src) for x in c.children if x.type not in ("throws", ",")]
            elif c.type == "default":
                continue
            elif c.type in ("identifier", "formal_parameters", "throws", "block", ";", "default_value"):
                if c.type == "default_value":
                    default_value = _text(c, self._src)
                continue
            elif not return_type:
                return_type = _text(c, self._src)

        return Method(
            name=_text(name_node, self._src),
            return_type=return_type,
            parameters=params,
            modifiers=mods,
            throws=throws,
            type_parameters=type_params,
            default_value=default_value,
            line=_line(node),
        )

    def _param_text(self, node: Node) -> str:
        """形参文本：类型 + 名字（含数组维度 / 可变参数）。"""
        if node.type == "spread_parameter":
            # 可变参数：type + "..." + variable_declarator(identifier)
            base = _first_child(node, "integral_type", "floating_point_type", "type_identifier",
                                "generic_type", "array_type", "scoped_type_identifier")
            name_node = None
            for c in node.children:
                if c.type == "variable_declarator":
                    name_node = _first_child(c, "identifier")
            base_text = _text(base, self._src)
            return f"{base_text}... {_text(name_node, self._src)}".strip()

        parts: List[str] = []
        for c in node.children:
            if c.type == "identifier":
                parts.append(_text(c, self._src))
            elif c.type == "modifiers":
                continue
            else:
                parts.append(_text(c, self._src))
        return " ".join(p for p in parts if p)

    def _parse_constructor(self, node: Node) -> Constructor:
        mods = self._parse_modifiers(_first_child(node, "modifiers"))
        name_node = _first_child(node, "identifier")
        params: List[str] = []
        throws: List[str] = []
        for c in node.children:
            if c.type == "formal_parameters":
                params = [self._param_text(p) for p in c.children
                          if p.type in ("formal_parameter", "spread_parameter")]
            elif c.type == "throws":
                throws = [_text(x, self._src) for x in c.children if x.type not in ("throws", ",")]
        return Constructor(
            name=_text(name_node, self._src),
            parameters=params,
            modifiers=mods,
            throws=throws,
            line=_line(node),
        )

    # ---------------- 类型声明 ----------------

    def _parse_type_decl(self, node: Node) -> TypeDecl:
        kind = TYPE_DECL_NODES[node.type]
        name_node = _first_child(node, "identifier")
        mods = self._parse_modifiers(_first_child(node, "modifiers"))

        td = TypeDecl(
            kind=kind,
            name=_text(name_node, self._src),
            modifiers=mods,
            line=_line(node),
            end_line=node.end_point[0] + 1,
        )

        # extends：普通类是 superclass，接口/记录是 extends_interfaces
        superclass_node = _first_child(node, "superclass")
        if superclass_node is not None:
            td.superclass = self._parse_superclass(superclass_node)
        else:
            ext = _first_child(node, "extends_interfaces")
            if ext is not None:
                td.extends_interfaces.extend(self._parse_type_list(ext))

        # implements
        impl = _first_child(node, "super_interfaces")
        if impl is not None:
            td.interfaces.extend(self._parse_type_list(impl))

        # record 组件
        if node.type == "record_declaration":
            fp = _first_child(node, "formal_parameters")
            if fp is not None:
                td.record_components = [
                    self._param_text(p) for p in fp.children
                    if p.type in ("formal_parameter", "spread_parameter")
                ]

        body = _first_child(node, "class_body", "interface_body", "enum_body", "annotation_type_body")
        if body is not None:
            self._fill_body(td, body)

        return td

    def _fill_body(self, td: TypeDecl, body: Node) -> None:
        """填充类型体中的字段 / 方法 / 嵌套类型 / 局部类。"""
        for child in body.children:
            ctype = child.type

            if ctype == "field_declaration":
                td.fields.extend(self._parse_field(child))
            elif ctype == "constant_declaration":
                # interface 内字段：隐式 public static final
                td.fields.extend(self._parse_field(child, implicit=["public", "static", "final"]))
            elif ctype == "method_declaration":
                td.methods.append(self._parse_method(child))
                self._collect_local_types(td, child)
            elif ctype == "annotation_type_element_declaration":
                td.methods.append(self._parse_method(child))
            elif ctype == "constructor_declaration":
                td.constructors.append(self._parse_constructor(child))
            elif ctype == "compact_constructor_declaration":
                td.constructors.append(self._parse_constructor(child))
            elif ctype == "enum_constant":
                cn = _first_child(child, "identifier")
                td.enum_constants.append(_text(cn, self._src))
                td.fields.append(Field(
                    name=_text(cn, self._src),
                    type="",
                    modifiers=Modifiers(keywords=["public", "static", "final"]),
                    is_enum_constant=True,
                    line=_line(child),
                ))
                # 枚举常量专属类体
                cb = _first_child(child, "class_body")
                if cb is not None:
                    anon = TypeDecl(kind="anonymous_class", name=f"{td.name}.{_text(cn, self._src)}")
                    self._fill_body(anon, cb)
                    td.nested_types.append(anon)
            elif ctype == "enum_body_declarations":
                self._fill_body(td, child)
            elif ctype in TYPE_DECL_NODES:
                td.nested_types.append(self._parse_type_decl(child))

    def _collect_local_types(self, td: TypeDecl, method_node: Node) -> None:
        """在方法体内查找局部类与匿名类。"""
        block = _first_child(method_node, "block")
        if block is None:
            return
        stack = list(block.children)
        while stack:
            n = stack.pop(0)
            if n.type in TYPE_DECL_NODES:
                td.nested_types.append(self._parse_type_decl(n))
                continue  # 不再深入该声明内部（其嵌套类由 _parse_type_decl 处理）
            if n.type == "object_creation_expression":
                cb = _first_child(n, "class_body")
                if cb is not None:
                    anon = TypeDecl(kind="anonymous_class", name="<anonymous>")
                    self._fill_body(anon, cb)
                    td.nested_types.append(anon)
            stack.extend(n.children)

    # ---------------- 顶层入口 ----------------

    def parse_bytes(self, data: bytes, path: str = "") -> JavaFile:
        self._src = data
        tree = self._parser.parse(data)
        root = tree.root_node

        result = JavaFile(path=path)
        result.errors = self._collect_errors(root)

        for child in root.children:
            if child.type == "package_declaration":
                ident = _first_child(child, "identifier", "scoped_identifier")
                result.package = _text(ident, self._src)
            elif child.type == "import_declaration":
                ident = _first_child(child, "identifier", "scoped_identifier")
                imp = _text(ident, self._src)
                if imp:
                    result.imports.append(imp)
            elif child.type in TYPE_DECL_NODES:
                result.types.append(self._parse_type_decl(child))

        return result

    def _collect_errors(self, node: Node) -> List[str]:
        errs: List[str] = []
        stack = [node]
        while stack:
            n = stack.pop()
            if n.type == "ERROR" or n.is_missing:
                snippet = self._src[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
                errs.append(f"line {n.start_point[0] + 1}: {snippet[:60]!r}")
            stack.extend(n.children)
        return errs

    def parse_file(self, path: str) -> JavaFile:
        with open(path, "rb") as f:
            data = f.read()
        return self.parse_bytes(data, path=path)

    def parse_source(self, source: str) -> JavaFile:
        return self.parse_bytes(source.encode("utf-8"), path="<memory>")


# --------------------------------------------------------------------------
# 序列化 / 输出
# --------------------------------------------------------------------------

def to_dict(obj: Any) -> Any:
    """dataclass -> 可 JSON 序列化的 dict（含派生属性）。"""
    if hasattr(obj, "__dataclass_fields__"):
        out: Dict[str, Any] = {}
        for k, v in obj.__dict__.items():
            if k.startswith("_"):
                continue
            out[k] = to_dict(v)
        if isinstance(obj, Field):
            out["is_static"] = obj.is_static
            out["is_final"] = obj.is_final
            out["is_constant"] = obj.is_constant
        elif isinstance(obj, TypeDecl):
            out["constants"] = [to_dict(f) for f in obj.constants]
        elif isinstance(obj, Method):
            out["signature"] = obj.signature
        elif isinstance(obj, Constructor):
            out["signature"] = obj.signature
        return out
    if isinstance(obj, list):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    return obj


def format_text(result: JavaFile, constants_only: bool = False) -> str:
    lines: List[str] = []
    lines.append("=" * 74)
    lines.append(f"文件: {result.path}")
    lines.append(f"包名: {result.package or '(default)'}")
    if result.imports:
        lines.append(f"导入  : {', '.join(result.imports)}")
    if result.errors:
        lines.append(f"⚠ 语法问题: {'; '.join(result.errors[:5])}")
    lines.append("=" * 74)

    for td in result.types:
        lines.extend(_format_type(td, indent=0, constants_only=constants_only))
    return "\n".join(lines)


def _format_type(td: TypeDecl, indent: int, constants_only: bool = False) -> List[str]:
    pad = "  " * indent
    out: List[str] = []

    anns = " ".join(str(a) for a in td.annotations)
    mods = " ".join(td.modifiers.keywords)
    head = " ".join(x for x in (anns, mods, td.kind, td.name) if x)
    out.append(f"{pad}{head}")

    if td.superclass:
        out.append(f"{pad}  └─ extends {td.superclass}")
    if td.extends_interfaces:
        out.append(f"{pad}  └─ extends {', '.join(td.extends_interfaces)}")
    if td.interfaces:
        out.append(f"{pad}  └─ implements {', '.join(td.interfaces)}")
    if td.record_components:
        out.append(f"{pad}  └─ record components: ({', '.join(td.record_components)})")
    if td.enum_constants:
        out.append(f"{pad}  └─ enum constants: {', '.join(td.enum_constants)}")

    if td.fields:
        out.append(f"{pad}  字段:")
        for f in td.fields:
            if constants_only and not f.is_constant:
                continue
            flags = " ".join(f.modifiers.keywords + f.implicit_modifiers)
            decl = " ".join(x for x in (flags, f.type, f.name) if x)
            line = f"{pad}    - {decl}"
            if f.value:
                line += f" = {f.value}"
            if f.is_constant:
                line += "   [static final]"
            elif f.is_static:
                line += "   [static]"
            elif f.is_final:
                line += "   [final]"
            out.append(line)

    if not constants_only:
        if td.constructors:
            out.append(f"{pad}  构造器:")
            for c in td.constructors:
                out.append(f"{pad}    - {c.signature}")
        if td.methods:
            out.append(f"{pad}  方法:")
            for m in td.methods:
                out.append(f"{pad}    - {m.signature}")

    for nested in td.nested_types:
        out.append("")
        out.extend(_format_type(nested, indent + 1, constants_only))

    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def iter_java_files(targets: List[str]) -> Iterator[str]:
    for target in targets:
        if os.path.isfile(target):
            yield target
            continue
        if not os.path.isdir(target):
            print(f"[WARN] 路径不存在: {target}", file=sys.stderr)
            continue
        for root, _dirs, files in os.walk(target):
            for f in sorted(files):
                if f.endswith(".java"):
                    yield os.path.join(root, f)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="tree-sitter Java 解析器")
    ap.add_argument("targets", nargs="+", help="Java 文件或目录（可多个）")
    ap.add_argument("--json", action="store_true", help="输出 JSON")
    ap.add_argument("--out", help="输出到文件")
    ap.add_argument("--constants-only", action="store_true", help="仅显示 static final 常量")
    args = ap.parse_args(argv)

    parser = JavaParser()
    results: List[JavaFile] = []
    seen = set()
    for path in iter_java_files(args.targets):
        if path in seen:
            continue
        seen.add(path)
        try:
            results.append(parser.parse_file(path))
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {path}: {e}", file=sys.stderr)

    if args.json:
        payload = json.dumps([to_dict(r) for r in results], ensure_ascii=False, indent=2)
    else:
        payload = "\n\n".join(format_text(r, args.constants_only) for r in results)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"已写入 {args.out}（{len(results)} 个文件）", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
