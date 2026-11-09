"""Learning curves for (P1) on the T-maze."""

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


BLUE = "#285F88"
COPPER = "#C87539"

INK = "#33383B"
MUTED = "#7F878D"
GRID = "#E8EAEB"
REFERENCE = "#B4BABF"


SERIES = (
    (
        (2, 0.9),
        BLUE,
        "-",
        r"$|\mathcal{Z}|=2,\ \alpha=0.9$"
    ),
    (
        (2, 1.0),
        BLUE,
        "--",
        r"$|\mathcal{Z}|=2,\ \alpha=1$"
    ),
    (
        (1, 0.9),
        COPPER,
        "-",
        r"$|\mathcal{Z}|=1,\ \alpha=0.9$"
    ),
    (
        (1, 1.0),
        COPPER,
        "--",
        r"$|\mathcal{Z}|=1,\ \alpha=1$"
    ),
)


def set_style(font_scale=1.0):
    fs = float(font_scale)

    plt.rcParams.update({

        "font.family": "serif",
        "font.serif": [
            "STIXGeneral",
            "DejaVu Serif"
        ],
        "mathtext.fontset": "stix",

        "font.size": 7.4 * fs,
        "axes.labelsize": 7.5 * fs,
        "xtick.labelsize": 6.5 * fs,
        "ytick.labelsize": 6.5 * fs,
        "legend.fontsize": 6.6 * fs,

        "axes.linewidth": 0.65,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,

        "axes.spines.top": False,
        "axes.spines.right": False,

        "xtick.color": MUTED,
        "ytick.color": MUTED,

        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,

        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        "axes.grid": False,

        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",

        "legend.frameon": False,

        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def load_runs(main_root, slack, margin):
    runs = {}

    directory = Path(main_root) / "tmaze_main"

    for result_path in sorted(
        directory.glob("*/result.json")
    ):
        log_path = result_path.parent / "log.json"

        if not log_path.exists():
            continue

        try:
            results = json.loads(
                result_path.read_text(encoding="utf-8")
            )

            logs = json.loads(
                log_path.read_text(encoding="utf-8")
            )

        except (OSError, ValueError) as exc:
            print(
                f"Skipping {result_path.parent}: {exc}"
            )

            continue

        for record in results.get("records", []):
            if record.get("variant") != "logic":
                continue

            if record.get("objective") != "P1":
                continue

            if record.get("slack") != slack:
                continue

            if record.get("margin") != margin:
                continue

            if isinstance(logs, dict):
                level = record.get("level")

                history = logs.get(
                    level,
                    logs.get(str(level))
                )

            else:
                history = logs

            if not isinstance(history, list):
                continue

            sequence = [
                float(step["J_phi"])
                for step in history
                if (
                    isinstance(step, dict)
                    and step.get("J_phi") is not None
                )
            ]

            if not sequence:
                continue

            key = (
                int(record["L"]),
                int(record["Z"]),
                float(record["alpha"])
            )

            runs.setdefault(key, []).append(sequence)

    return runs


def pad_histories(sequences, length):
    matrix = np.empty(
        (len(sequences), length),
        dtype=float
    )

    for idx, sequence in enumerate(sequences):
        m = len(sequence)

        matrix[idx, :m] = sequence

        matrix[idx, m:] = sequence[-1]

    return matrix


def band_limits(matrix, kind):
    if kind == "minmax":
        lower = matrix.min(axis=0)
        upper = matrix.max(axis=0)

    elif kind == "p10-p90":
        lower = np.percentile(
            matrix, 10, axis=0
        )

        upper = np.percentile(
            matrix, 90, axis=0
        )

    elif kind == "iqr":
        lower = np.percentile(
            matrix, 25, axis=0
        )

        upper = np.percentile(
            matrix, 75, axis=0
        )

    else:
        return None

    return lower, upper


def save_figure(fig, out_dir, basename, dpi):
    out_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    formats = (
        ("pdf", {}),
        ("png", {"dpi": dpi})
    )

    for extension, kwargs in formats:
        filename = (
            out_dir / f"{basename}.{extension}"
        )

        try:
            fig.savefig(
                filename,
                **kwargs
            )

        except OSError:
            filename = (
                out_dir / f"{basename}_new.{extension}"
            )

            fig.savefig(
                filename,
                **kwargs
            )

        print(f"Wrote {filename}")


def plot(runs, lengths, args):
    selected = [
        L for L in lengths
        if any(
            key[0] == L
            and key[1] in (1, 2)
            and key[2] in (0.9, 1.0)
            for key in runs
        )
    ]

    if not selected:
        raise SystemExit(
            "No matching P1 T-maze runs found.\n"
            f"Root: {args.main}\n"
            f"Slack: {args.slack}\n"
            f"Margin: {args.margin}"
        )

    set_style(args.font_scale)


    n_sweeps = max(
        len(seq)
        for key, histories in runs.items()
        if (
            key[0] in selected
            and key[1] in (1, 2)
            and key[2] in (0.9, 1.0)
        )
        for seq in histories
    )


    fig, axes = plt.subplots(
        1,
        len(selected),
        figsize=(args.width, args.height),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    axes = axes[0]

    x = np.arange(n_sweeps)


    for ax, L in zip(axes, selected):
        ax.set_axisbelow(True)

        ax.grid(
            axis="y",
            color=GRID,
            linewidth=0.55,
            zorder=0
        )


        if (
            args.margin == 0
            and not args.no_ceilings
        ):
            for ceiling in (0.5, 1.0):
                ax.axhline(
                    ceiling,
                    color=REFERENCE,
                    linestyle=(0, (2, 3)),
                    linewidth=0.85,
                    zorder=1
                )


        for (
            (Z, alpha),
            color,
            linestyle,
            _
        ) in SERIES:
            sequences = runs.get(
                (L, Z, alpha),
                []
            )

            if not sequences:
                continue

            matrix = pad_histories(
                sequences,
                n_sweeps
            )


            for seed in matrix:
                ax.plot(
                    x,
                    seed,
                    color=color,
                    linestyle="-",
                    linewidth=0.55,
                    alpha=args.seed_alpha,
                    zorder=2,
                    rasterized=True
                )


            limits = band_limits(
                matrix,
                args.band
            )

            if limits is not None:
                lower, upper = limits

                ax.fill_between(
                    x,
                    lower,
                    upper,
                    color=color,
                    alpha=args.band_alpha,
                    linewidth=0,
                    zorder=2
                )


            median = np.median(
                matrix,
                axis=0
            )

            ax.plot(
                x,
                median,
                linestyle=linestyle,
                color=color,
                linewidth=1.85,
                zorder=5
            )


            stopping_sweep = int(
                np.median([
                    len(seq) - 1
                    for seq in sequences
                ])
            )

            ax.plot(
                stopping_sweep,
                median[stopping_sweep],

                marker="o",
                markersize=4.1,

                markerfacecolor="white",
                markeredgecolor=color,
                markeredgewidth=1.1,

                zorder=7
            )


        ax.text(
            0.05,
            0.95,
            rf"$L={L}$",

            transform=ax.transAxes,

            ha="left",
            va="top",

            fontsize=8.2 * args.font_scale,
            color=INK,

            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.80,
                "pad": 1.2
            },

            zorder=10
        )


        ax.set_ylim(
            -0.025,
            1.06
        )

        ax.set_yticks([
            0.0,
            0.25,
            0.5,
            0.75,
            1.0
        ])

        ax.set_xlim(
            -0.2,
            max(n_sweeps - 1, 1) + 0.2
        )

        ax.xaxis.set_major_locator(
            MaxNLocator(
                nbins=4,
                integer=True
            )
        )


    fig.supxlabel(
        "Policy-improvement sweep",
        x=0.5,
        y=0.16,
        fontsize=7.5 * args.font_scale,
        color=INK
    )

    fig.supylabel(
        r"Satisfaction probability $J_\varphi$",
        x=0.015,
        fontsize=7.5 * args.font_scale,
        color=INK
    )


    handles = [

        Line2D(
            [0],
            [0],

            color=color,
            linestyle=linestyle,
            linewidth=1.85,

            label=label
        )

        for _, color, linestyle, label in SERIES
    ]

    fig.legend(
        handles=handles,

        loc="lower center",
        ncol=4,

        bbox_to_anchor=(0.5, 0.025),

        frameon=False,

        columnspacing=1.35,
        handlelength=2.05,
        handletextpad=0.5,
        borderpad=0.0,

        fontsize=6.65 * args.font_scale
    )


    fig.subplots_adjust(
        left=0.095,
        right=0.99,

        top=0.94,
        bottom=0.32,

        wspace=0.18
    )


    save_figure(
        fig,
        Path(args.out),
        args.name,
        args.dpi
    )

    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )


    parser.add_argument(
        "--main",
        default="runs_main",
        help="Root containing tmaze_main/"
    )

    parser.add_argument(
        "--out",
        default="reports_paper",
        help="Output directory"
    )

    parser.add_argument(
        "--slack",
        type=int,
        default=2
    )

    parser.add_argument(
        "--margin",
        type=int,
        default=0
    )

    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[3, 5, 10, 20]
    )

    parser.add_argument(
        "--name",
        default="fig4_tmaze_seeds"
    )


    parser.add_argument(
        "--band",

        choices=[
            "minmax",
            "p10-p90",
            "iqr",
            "none"
        ],

        default="minmax",

        help="Shaded uncertainty band"
    )


    parser.add_argument(
        "--width",
        type=float,
        default=7.1,
        help="Figure width in inches"
    )

    parser.add_argument(
        "--height",
        type=float,
        default=2.7,
        help="Figure height in inches"
    )


    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.0,
        help="Multiply all font sizes by this factor"
    )


    parser.add_argument(
        "--seed-alpha",
        type=float,
        default=0.055,
        help="Opacity of individual seed trajectories"
    )


    parser.add_argument(
        "--band-alpha",
        type=float,
        default=0.075,
        help="Opacity of uncertainty envelopes"
    )


    parser.add_argument(
        "--dpi",
        type=int,
        default=300
    )


    parser.add_argument(
        "--no-ceilings",
        action="store_true",
        help="Hide theoretical reference lines"
    )

    args = parser.parse_args(argv)


    runs = load_runs(
        args.main,
        args.slack,
        args.margin
    )


    plot(
        runs,
        args.lengths,
        args
    )


if __name__ == "__main__":
    main()
