import argparse
import asyncio

from tests.evals.eval_e2e_graph_building import build_baseline_graph, eval_graph


async def main():
    parser = argparse.ArgumentParser(
        description='Run eval_graph and optionally build_baseline_graph from the command line.'
    )

    parser.add_argument(
        '--multi-session-count',
        type=int,
        required=True,
        help='Integer representing multi-session count',
    )
    parser.add_argument('--session-length', type=int, required=True, help='Length of each session')
    parser.add_argument(
        '--build-baseline', action='store_true', help='If set, also runs build_baseline_graph'
    )

    args = parser.parse_args()

    # Optionally run the async function
    if args.build_baseline:
        print('Running build_baseline_graph...')
        await build_baseline_graph(
            multi_session_count=args.multi_session_count, session_length=args.session_length
        )

    # Always call eval_graph
    result = await eval_graph(
        multi_session_count=args.multi_session_count, session_length=args.session_length
    )
    print('Result of eval_graph:', result)


if __name__ == '__main__':
    asyncio.run(main())
