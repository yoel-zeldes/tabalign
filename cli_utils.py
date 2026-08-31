import argparse


def parse_student_n(val_str):
    try:
        val = float(val_str)
    except (ValueError, TypeError):
        raise argparse.ArgumentTypeError(f"Invalid student_n value: '{val_str}'. Must be a number.")

    if val < 0:
        if val != -1:
            raise argparse.ArgumentTypeError(
                f"Invalid student_n value: {val_str}. Negative values must be -1 to indicate full dataset."
            )
        return int(val)
    elif 0 < val < 1:
        return val
    elif val >= 1:
        if not val.is_integer():
            raise argparse.ArgumentTypeError(
                f"Invalid student_n value: {val_str}. Values >= 1 must be integers."
            )
        return int(val)
    else:
        raise argparse.ArgumentTypeError(
            f"Invalid student_n value: {val_str}. Must be an integer >= 1, a fraction between 0 and 1, or -1."
        )
