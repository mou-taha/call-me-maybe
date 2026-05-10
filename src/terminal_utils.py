
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
RESET = "\033[0m"
RED = "\033[31m"

HEADER = (f"{RED}{'=' * 80}\n" + "      CALL-ME-MAYBE      \n"
          + f"{'=' * 80}{RESET}\n")


def print_status(message: str, color: str = BLUE) -> None:
    """a helper function to print status messages in
    different colors for better visibility."""
    print(f"{color}{message}{RESET}")


def clear_terminal() -> None:
    """a helper function to clear the terminal screen."""
    print("\033[2J\033[H", end="")
    print(HEADER, end="")
