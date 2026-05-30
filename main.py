"""
main.py
-------
Entry point for running PUF attacks from the command line.

Usage
-----
    python main.py xor    -- run the XOR PUF evolutionary attack
    python main.py ipuf   -- run the Interpose PUF evolutionary attack
"""

import sys

from Attack.FunAttack_v2 import xor_attk, ipuf_attk


def main() -> None:
    """Parse the attack type argument and dispatch to the appropriate attack."""
    if len(sys.argv) != 2:
        print("Usage: python main.py <attack>")
        print("Choose 'xor' for XOR PUF attack or 'ipuf' for iPUF attack.")
        return

    attack_type = sys.argv[1].lower()

    if attack_type == "xor":
        res = xor_attk()
        print(res)
    elif attack_type == "ipuf":
        res = ipuf_attk()
        print(res)
    else:
        print("Invalid attack type. Choose 'xor' or 'ipuf'.")


if __name__ == "__main__":
    main()
