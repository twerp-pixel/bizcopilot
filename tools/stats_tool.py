"""Small stats helpers the agent can call for computation steps (no LLM math)."""
import statistics


def pct_change(old, new):
    if old == 0:
        return None
    return round((new - old) / old * 100, 2)


def zscores(values):
    if len(values) < 2:
        return [0.0 for _ in values]
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values) or 1e-8
    return [round((v - mean) / stdev, 2) for v in values]


def flag_anomalies(values, threshold=-1.5):
    """Return indices where z-score drops below threshold (a meaningful dip)."""
    z = zscores(values)
    return [i for i, s in enumerate(z) if s <= threshold]


if __name__ == "__main__":
    series = [220, 215, 230, 225, 218, 95, 228]
    print("pct_change week5->week6:", pct_change(series[4], series[5]))
    print("zscores:", zscores(series))
    print("anomalies at index:", flag_anomalies(series))
