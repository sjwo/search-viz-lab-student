import altair as alt
import pandas as pd


def generate(args):
    lib.Problem.generate(args.n_rows, args.n_cols, args.p_blocked, args.seed)


def search(args):
    problem = lib.Problem()
    closed = lib.ClosedList()
    open = None
    expansions = 0

    # how are we going to search through the state space?
    match args.order:
        case "dfs":
            raise NotImplementedError("open = ???")
        case "bfs":
            raise NotImplementedError("open = ???")
        case "gbfs":
            raise NotImplementedError("open = ???")
        case _:
            raise RuntimeError("unrecognized order")

    # note where we're starting, so we don't go in circles
    closed.insert(problem.start)

    start_node = lib.Node(problem.start)

    if args.order == "gbfs":
        raise NotImplementedError("compute start state heuristic value")
        raise NotImplementedError("label start node with heuristic value")

    open.insert(start_node)
    while open:
        n = open.extract()

        # did we find the goal?
        if n.state == problem.goal:
            solution_path = n.recover_ancestry()
            solution_cost = len(solution_path) - 1
            lib.log(f"cost: {solution_cost}")
            lib.log(f"expansions: {expansions}")
            if args.log:
                lib.log_solution(solution_path)
            quit()

        # where can we go from here?
        children = lib.expand(n.state, problem)
        expansions += 1
        if args.log:
            lib.log_expansion(n.state, expansions)

        for child_state in children:
            # let's make sure we haven't gone there already (don't go in circles)
            if child_state not in closed:
                closed.insert(child_state)
                child_node = lib.Node(child_state)
                child_node.parent = n
                if args.order == "gbfs":
                    raise NotImplementedError("compute state heuristic value")
                    raise NotImplementedError("label node with heuristic value")
                open.insert(child_node)

    lib.log("no solution found")
    lib.log(f"expansions: {expansions}")


def visualize(args):
    (n_rows, n_cols, start, goal, blocked, expansions, solution) = lib.read_log()
    expansions = pd.DataFrame(expansions)
    blocked = pd.DataFrame(blocked, columns=["x", "y"])
    start = pd.DataFrame([start], columns=["x", "y"])
    goal = pd.DataFrame([goal], columns=["x", "y"])

    grid_chart = (
        alt.Chart(blocked)
        .mark_rect(color="black")
        .encode(
            x=alt.X("x:O", scale=alt.Scale(domain=list(range(0, n_cols)))),
            y=alt.Y(
                "y:O", scale=alt.Scale(reverse=True, domain=list(range(0, n_rows)))
            ),
        )
    )
    expansions_overlay = (
        alt.Chart(expansions)
        .mark_rect()
        .encode(
            x=alt.X("x:O", scale=alt.Scale(domain=list(range(0, n_cols)))),
            y=alt.Y(
                "y:O", scale=alt.Scale(reverse=True, domain=list(range(0, n_rows)))
            ),
            color="expansion:Q",
        )
    )
    start_overlay = (
        alt.Chart(start)
        .mark_point(color="green")
        .encode(
            x="x:O",
            y="y:O",
        )
    )
    goal_overlay = (
        alt.Chart(goal)
        .mark_point(color="red")
        .encode(
            x="x:O",
            y="y:O",
        )
    )
    chart = grid_chart + expansions_overlay + start_overlay + goal_overlay
    lib.log(f"saving plot to '{args.out}'")
    chart.save(args.out)


def main():
    args = lib.parse_args()
    match args.command:
        case "generate":
            generate(args)
        case "search":
            if args.log:
                print("x,y,expansion")
            search(args)
        case "visualize":
            visualize(args)
        case _:
            raise RuntimeError(f"unknown command {args.command}")


if __name__ == "__main__":
    main()
