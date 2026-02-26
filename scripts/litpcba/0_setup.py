from src.chem import LITPCBA_RECEPTORS, UniDocker


def main():
    for rec in LITPCBA_RECEPTORS:
        UniDocker(rec)


if __name__ == "__main__":
    main()
