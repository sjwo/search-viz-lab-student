import argparse
import collections
import heapq as hq
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Self

##########################
# SEARCH ALGORITHM STUFF #
##########################


class Node:
    generations = 0

    def __init__(self, state, parent: Self = None):
        self.state: State = state
        self.h: float = None
        self.parent = parent
        Node.generations += 1
        self.genid = Node.generations

    def recover_ancestry(self):
        ancestry = list()
        cur = self
        while cur.parent is not None:
            ancestry.append((cur.state.x, cur.state.y))
            cur = cur.parent
        ancestry.append((cur.state.x, cur.state.y))
        ancestry.reverse()
        return ancestry

    def __lt__(self, other):
        self.genid < other.genid

    def __gt__(self, other):
        self.genid > other.genid

    def __eq__(self, other):
        self.genid == other.genid


class ClosedList:
    def __init__(self):
        self._data = set()

    def insert(self, elt):
        self._data.add(elt)

    def contains(self, elt):
        return elt in self._data

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        return iter(self._data)


class OpenList(ABC):
    @abstractmethod
    def insert(self):
        pass

    @abstractmethod
    def extract(self):
        pass

    @abstractmethod
    def __len__(self):
        pass


class Stack(OpenList):
    def __init__(self):
        self._data = list()

    def insert(self, node):
        raise NotImplementedError()

    def extract(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self._data)


class Queue(OpenList):
    def __init__(self):
        self._data = collections.deque()

    def insert(self, node):
        raise NotImplementedError("read documentation!")

    def extract(self):
        raise NotImplementedError("read documentation!")

    def __len__(self):
        return len(self._data)


class GBFSPriorityQueue(OpenList):
    def __init__(self):
        self._data = list()

    def insert(self, node):
        hq.heappush(self._data, (node.h, node))

    def extract(self):
        (key, elt) = hq.heappop(self._data)
        return elt

    def __len__(self):
        return len(self._data)


########################
# SEARCH PROBLEM STUFF #
########################


@dataclass
class State:
    x: int
    y: int

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

    def __repr__(self):
        return f"({self.x}, {self.y})"


def manhattan_distance(a: State, b: State):
    return abs(a.x - b.x) + abs(a.y - b.y)


class Problem:
    def read_grid(n_rows, n_cols):
        def row_to_y(row, n_rows):
            """0-origin row id, 0-origin y-value. 0th row at top; y=0 at bottom"""
            return n_rows - row - 1

        blocked = set()
        for row in range(n_rows):
            line = sys.stdin.readline().strip()
            y = row_to_y(row, n_rows)
            for x in range(len(line)):
                cell = line[x]
                match cell:
                    # blocked cell
                    case "#":
                        blocked.add((x, y))
                    # empty space
                    case "_":
                        pass
                    # start
                    case "^":
                        start = State(x, y)
                    # goal
                    case "*":
                        goal = State(x, y)
        return (start, goal, blocked)

    def __init__(self):
        """new from stdin"""

        n_rows = int(sys.stdin.readline().strip())
        n_cols = int(sys.stdin.readline().strip())
        (start, goal, blocked) = Problem.read_grid(n_rows, n_cols)
        self.blocked = blocked
        self.max_x = n_cols - 1
        self.max_y = n_rows - 1
        self.start = start
        self.goal = goal

    def generate(
        n_rows,
        n_cols,
        p_blocked,
        seed=None,
    ) -> Self:
        """instance not guaranteed to be solvable!

        start, goal are left and right sides at mid-height"""
        if seed is None:
            seed = random.randrange(sys.maxsize)
        log(f"seed: {seed}")
        random.seed(seed)

        rows = list()
        for r in range(n_rows):
            rows.append(list())
            for c in range(n_cols):
                cell = "_" if random.random() > p_blocked else "#"
                rows[-1].append(cell)

        midheight = int(n_rows / 2)
        rows[midheight][0] = "^"
        rows[midheight][-1] = "*"

        # write to stdout
        print(n_rows)
        print(n_cols)
        for row in rows:
            for cell in row:
                print(cell, end="")
            print("\n", end="")


def expand(state: State, problem: Problem):
    # ensure we have a valid state
    assert 0 <= state.x <= problem.max_x
    assert 0 <= state.y <= problem.max_y
    assert (state.x, state.y) not in problem.blocked, (
        f"trying to expand blocked state: {(state.x, state.y)}"
    )

    successors = list()

    # west
    if state.x - 1 >= 0 and (state.x - 1, state.y) not in problem.blocked:
        successors.append(State(state.x - 1, state.y))
    # south
    if state.y - 1 >= 0 and (state.x, state.y - 1) not in problem.blocked:
        successors.append(State(state.x, state.y - 1))
    # north
    if state.y + 1 <= problem.max_y and (state.x, state.y + 1) not in problem.blocked:
        successors.append(State(state.x, state.y + 1))
    # east
    if state.x + 1 <= problem.max_x and (state.x + 1, state.y) not in problem.blocked:
        successors.append(State(state.x + 1, state.y))
    # log(f"successors: {successors}")
    return successors


#######################
# VISUALIZATION STUFF #
#######################


def read_log():
    """assumes all values in csv are ints"""
    n_rows = int(sys.stdin.readline().strip())
    n_cols = int(sys.stdin.readline().strip())
    (start, goal, blocked) = Problem.read_grid(n_rows, n_cols)
    header = list(sys.stdin.readline().strip().split(","))
    expansions = {key: list() for key in header}

    while line := sys.stdin.readline().strip():
        if line == "solution:":
            break
        parts = line.split(",")
        assert len(header) == len(parts)
        for i in range(len(header)):
            expansions[header[i]].append(int(parts[i]))
    solution_line = sys.stdin.readline().strip()
    solution = eval(f"[{solution_line}]")
    return (n_rows, n_cols, start, goal, blocked, expansions, solution)


#################
# GENERAL STUFF #
#################


def log(msg):
    print(msg, file=sys.stderr)


def log_expansion(state, expansion_number):
    print(f"{state.x},{state.y},{expansion_number}")


def log_solution(solution_path):
    print("solution:")
    print(",".join([str(x) for x in solution_path]))


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="A state space search grid-world demo. Generate simple grid-world problems with static obstacles, solve them with a variety of graph search algorithms, and visualize the search behavior.",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        help="run a command with `--help` flag to learn more",
    )
    subparsers.required = True

    generate_help = "generates grid-world problem instance, printing to stdout, logging rng seed to stderr"
    parser_generate = subparsers.add_parser(
        "generate",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=generate_help,
        description=generate_help,
    )

    def set_generate(args):
        args.command = "generate"

    parser_generate.set_defaults(func=set_generate)
    parser_generate.add_argument("n_rows", type=int, help="number of rows in problem,")
    parser_generate.add_argument(
        "n_cols", type=int, help="number of columns in problem,"
    )
    parser_generate.add_argument(
        "p_blocked", type=float, help="probability that any given cell is blocked,"
    )
    parser_generate.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed for random number generator (for reproducibility)",
    )

    search_help = "reads problem instance from stdin, runs state space search from start to goal using specified node ordering. reports results on stderr"
    parser_search = subparsers.add_parser(
        "search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=search_help,
        description=search_help,
    )

    def set_search(args):
        args.command = "search"

    parser_search.set_defaults(func=set_search)
    parser_search.add_argument(
        "order",
        choices=["dfs", "bfs", "gbfs"],
        help="order in which to search through the problem state space",
    )
    parser_search.add_argument(
        "--log", action="store_true", help="log expansions to stdout"
    )

    visualize_help = "reads problem file followed by run log (expansions and solution) on stdin, saves expansion order plot to PDF."
    parser_visualize = subparsers.add_parser(
        "visualize",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        help=visualize_help,
        description=visualize_help,
    )

    def set_visualize(args):
        args.command = "visualize"

    parser_visualize.set_defaults(func=set_visualize)
    parser_visualize.add_argument("--out", default="out.pdf", help="output filename")
    args = parser.parse_args()
    args.func(args)
    return args
