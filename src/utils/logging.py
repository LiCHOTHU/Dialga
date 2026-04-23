import logging


def configure_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
    )


def log_metrics(logger, prefix, metrics):
    formatted = " | ".join(f"{key}: {value:.6f}" for key, value in metrics.items())
    logger.info("%s | %s", prefix, formatted)
