"""Plot command for visualizing benchmark results."""
import glob
import json
import os


def flatten(obj, prefix='', numeric_only=True):
    """Flatten a nested dict with dot-separated keys."""
    items = {}
    for k, v in obj.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten(v, key, numeric_only))
        elif not numeric_only or isinstance(v, (int, float)):
            items[key] = v
    return items


def execute(args):
    """Execute the plot command."""
    import matplotlib.pyplot as plt

    files = []
    if args.file:
        files = [args.file]
    elif args.directory:
        files = sorted(glob.glob(os.path.join(args.directory, '**', '*.json'), recursive=True))
        if not files:
            print(f"No .json files found in {args.directory}")
            return

    # Parse all JSONL records
    records = []
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

    if not records:
        print("No records found in JSON files.")
        return

    # Collect all keys across records
    all_keys = set()
    all_keys_any = set()
    for rec in records:
        all_keys.update(flatten(rec).keys())
        all_keys_any.update(flatten(rec, numeric_only=False).keys())

    keys = sorted(all_keys)
    keys_any = sorted(all_keys_any)
    if not keys:
        print("No numeric attributes found.")
        return

    # Use CLI flags if provided, otherwise prompt interactively
    if args.x and args.y:
        x_key = args.x
        y_key = args.y
        x_log = args.xlog
        y_log = args.ylog
        label_key = None
        if x_key not in all_keys:
            print(f"Unknown attribute: {x_key}")
            return
        if y_key not in all_keys:
            print(f"Unknown attribute: {y_key}")
            return
    else:
        print(f"\nFound {len(records)} records with {len(keys)} numeric attributes:\n")
        for i, key in enumerate(keys, 1):
            print(f"  {i:3d}. {key}")

        # Prompt for axis selections
        print()
        x_idx = int(input("Select X-axis attribute (number): ")) - 1
        x_key = keys[x_idx]
        x_log = input(f"  Log scale for X ({x_key})? [y/N]: ").strip().lower() == 'y'
        y_idx = int(input("Select Y-axis attribute (number): ")) - 1
        y_key = keys[y_idx]
        y_log = input(f"  Log scale for Y ({y_key})? [y/N]: ").strip().lower() == 'y'

        print(f"\nAll attributes (for labeling):\n")
        for i, key in enumerate(keys_any, 1):
            print(f"  {i:3d}. {key}")
        print()
        label_input = input("Select label attribute (number, or Enter to skip): ").strip()
        label_key = keys_any[int(label_input) - 1] if label_input else None

    # Extract paired values, grouped by label
    from collections import defaultdict
    groups = defaultdict(lambda: ([], []))
    for rec in records:
        flat = flatten(rec)
        flat_all = flatten(rec, numeric_only=False)
        if x_key in flat and y_key in flat:
            if label_key and label_key in flat_all:
                label = str(flat_all[label_key])
            else:
                label = 'unlabeled' if label_key else None
            groups[label][0].append(flat[x_key])
            groups[label][1].append(flat[y_key])

    if not groups:
        print(f"No records contain both '{x_key}' and '{y_key}'.")
        return

    total = sum(len(xs) for xs, _ in groups.values())
    print(f"\nPlotting {total} points: {x_key} vs {y_key}")

    fig, ax = plt.subplots()
    for label in sorted(groups, key=lambda l: (l is None, l)):
        xs, ys = groups[label]
        ax.scatter(xs, ys, alpha=0.7, label=label)
    if label_key:
        ax.legend(title=label_key)
    if x_log:
        ax.set_xscale('log')
    if y_log:
        ax.set_yscale('log')
    if x_log:
        ax.set_xticks(sorted(set(xs)))
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if y_log:
        ax.set_yticks(sorted(set(ys)))
        ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel(x_key)
    ax.set_ylabel(y_key)
    ax.set_title(f"{y_key} vs {x_key}")
    fig.tight_layout()

    if args.output:
        plt.savefig(args.output)
        print(f"Saved to {args.output}")
    else:
        plt.show()


def register_plot_command(subparsers):
    """Register the plot subcommand."""
    parser = subparsers.add_parser(
        'plot',
        help='Plot benchmark results from JSONL files'
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        '-d', '--directory',
        type=str,
        help='Path to a results directory containing .json files'
    )
    source.add_argument(
        '-f', '--file',
        type=str,
        help='Path to a single JSONL file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output image path (shows interactive window if not set)'
    )
    parser.add_argument(
        '-x',
        type=str,
        default=None,
        help='X-axis attribute (dot-separated key, e.g. num_vectors)'
    )
    parser.add_argument(
        '-y',
        type=str,
        default=None,
        help='Y-axis attribute (dot-separated key, e.g. index_creation_time_s)'
    )
    parser.add_argument(
        '--xlog',
        action='store_true',
        help='Use log scale for X-axis'
    )
    parser.add_argument(
        '--ylog',
        action='store_true',
        help='Use log scale for Y-axis'
    )
    parser.set_defaults(func=execute)
