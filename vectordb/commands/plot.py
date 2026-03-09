"""Plot command for visualizing benchmark results."""
import glob
import json
import math
import os

from matplotlib.ticker import MaxNLocator


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
        label_keys = args.label or []
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
        label_keys = [keys_any[int(label_input) - 1]] if label_input else []

    # Extract paired values, grouped by facet and label
    from collections import defaultdict
    facet_key = args.facet
    # label_keys minus the facet key (no point repeating it in the legend)
    series_keys = [k for k in label_keys if k != facet_key]

    faceted = defaultdict(lambda: defaultdict(lambda: ([], [])))
    for rec in records:
        flat = flatten(rec)
        flat_all = flatten(rec, numeric_only=False)
        if x_key not in flat or y_key not in flat:
            continue

        if facet_key and facet_key in flat_all:
            facet_val = str(flat_all[facet_key])
        else:
            facet_val = None

        if series_keys:
            parts = [f"{k}={flat_all[k]}" for k in series_keys if k in flat_all]
            label = ", ".join(parts) if parts else 'unlabeled'
        else:
            label = None

        faceted[facet_val][label][0].append(flat[x_key])
        faceted[facet_val][label][1].append(flat[y_key])

    if not faceted:
        print(f"No records contain both '{x_key}' and '{y_key}'.")
        return

    total = sum(
        len(xs)
        for groups in faceted.values()
        for xs, _ in groups.values()
    )
    print(f"\nPlotting {total} points: {x_key} vs {y_key}")

    facet_vals = sorted(faceted, key=lambda v: (v is None, v))
    ncols = min(len(facet_vals), 2)
    nrows = math.ceil(len(facet_vals) / ncols)

    fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                             sharex=True, sharey=True,
                             figsize=(6 * ncols, 4 * nrows))

    # Collect all series labels for consistent colors across panels
    all_labels = sorted({
        label
        for groups in faceted.values()
        for label in groups
        if label is not None
    })
    color_map = {label: f"C{i}" for i, label in enumerate(all_labels)}

    for idx, facet_val in enumerate(facet_vals):
        ax = axes[idx // ncols][idx % ncols]
        groups = faceted[facet_val]
        for label in sorted(groups, key=lambda l: (l is None, l)):
            xs, ys = groups[label]
            # Sort by x so line plots connect in order
            paired = sorted(zip(xs, ys))
            xs, ys = [p[0] for p in paired], [p[1] for p in paired]
            color = color_map.get(label)
            ax.plot(xs, ys, marker='o', alpha=0.7, label=label, color=color)
        if facet_key:
            ax.set_title(f"{facet_key}={facet_val}")
        if x_log:
            ax.set_xscale('log')
            ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.xaxis.set_major_formatter(plt.ScalarFormatter())
        ax.ticklabel_format(axis='x', style='plain')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        if y_log:
            ax.set_yscale('log')
            ax.yaxis.set_minor_locator(plt.NullLocator())
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
        ax.yaxis.set_major_formatter(plt.ScalarFormatter())
        ax.ticklabel_format(axis='y', style='plain')
        ax.set_xlabel(x_key)
        ax.set_ylabel(y_key)

    # Hide unused subplot slots
    for idx in range(len(facet_vals), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    title = f"{y_key} vs {x_key}"
    if args.title:
        title += f"\n{args.title}"
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.82, 0.95])

    # Shared legend — placed after tight_layout so it doesn't get clipped
    if series_keys:
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, title=", ".join(series_keys),
                   loc='center right', bbox_to_anchor=(0.98, 0.5))

    if args.output:
        plt.savefig(args.output, bbox_inches='tight')
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
    parser.add_argument(
        '-l', '--label',
        type=str,
        action='append',
        default=[],
        help='Attribute to group/label data points by (repeatable, e.g. -l m -l ef_construction)'
    )
    parser.add_argument(
        '--facet',
        type=str,
        default=None,
        help='Attribute to split into subplots (e.g. --facet m)'
    )
    parser.add_argument(
        '-t', '--title',
        type=str,
        default=None,
        help='Note to append below the auto-generated title'
    )
    parser.set_defaults(func=execute)
