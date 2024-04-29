from abc import ABC
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Type, TypedDict, Union
import torch.fx as fx

from ..lang.functional_types import Memory, Register
from .._support.indexing import IndexExpr
from .._support.dtype import DataType


def get_node_name(string: str, skip_first: bool = True):
    snakeString = ""
    if skip_first:
        snakeString += string[0].lower()
        string = string[1:]
    for i in string:
        if i.isupper():
            snakeString += "_" + i.lower()
        else:
            snakeString += i
    # Drop the "_node" suffix
    return snakeString[:-5]


@dataclass
class CustomNode(ABC):
    """
    Base class for all custom fx nodes.
    """

    op: Any

    @classmethod
    def from_fx_node(cls, node: fx.Node):
        instance = cls(node.graph, node.op, *node.args)
        instance.fx_node = node
        return instance

    def __str__(self) -> str:
        name = get_node_name(self.__class__.__name__)
        vars_list = [f"{key}={value}" for key, value in vars(self).items()][2:]
        vars_str = ", ".join(vars_list)
        return f"{name} {vars_str}" + "\n"

    def custom_string(self, value_map: dict[str, str]) -> str:
        # If a subclass does not define custom printing we revert to the default
        return str(self)

    def add_to_graph(self, graph: fx.Graph):
        arg_list = tuple([value for _, value in vars(self).items()][2:])
        self.fx_node = graph.create_proxy(
            "call_function",
            target=self.op,
            args=arg_list,
            kwargs={},
        )

    @classmethod
    def handle(cls, graph, *args, **kwargs):
        node = cls(*args, **kwargs)
        node.add_to_graph(graph)
        return node.fx_node

    @property
    def name(self) -> str:
        if hasattr(self, "_name"):
            return self._name
        return self.fx_node.name


@dataclass
class UnknownNode(CustomNode):
    """
    Represents an fx.Node that has no corresponding CustomNode class.
    """

    args: Sequence[Any]
    kwargs: dict[Any, Any]

    @classmethod
    def from_fx_node(cls, node: fx.Node):
        return cls(node.graph, node.op, node.args, node.kwargs)


@dataclass
class PlaceholderNode(CustomNode):
    """
    Represents a placeholder node in the graph, i.e. an input to a function.
    """

    _name: str
    type: Optional[DataType]

    @classmethod
    def from_fx_node(cls, node: fx.Node):
        return cls(node.graph, node.op, node.name, node.type)


# Nodes modeling TKL operations in the kernel language


@dataclass
class ConstructRegisterFromMetadataNode(CustomNode):
    shape: tuple[IndexExpr, ...]
    dtype: DataType
    value: float


@dataclass
class MmaNode(CustomNode):
    lhs: fx.Node
    rhs: fx.Node
    acc: fx.Node


@dataclass
class ReadNode(CustomNode):
    memory: fx.Proxy
    elements_per_thread: Optional[Any] = None
    type: Optional[Type[Register]] = None


@dataclass
class TiledLoop(CustomNode):
    axis: IndexExpr
    init_args: Sequence[Any]
    subgraph_name: str
    implicit_captures: Sequence[fx.Proxy]

    # def handle(self, graph, op, axis: IndexExpr, init_args=[]):
    #     def wrapper(f):
    #         with graph.subtracer() as subtracer:
    #             subgraph_name, implicit_capture = subtracer.trace(f)
    #         node = TiledLoop(op, axis, init_args, subgraph_name, implicit_capture)
    #         node.add_to_graph(self.region_graph)
    #         return node.fx_node


@dataclass
class WriteNode(CustomNode):
    register_: fx.Proxy
    memory: fx.Proxy
    elements_per_thread: Optional[Any]
