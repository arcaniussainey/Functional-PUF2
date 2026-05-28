from Attack.FunAttack import *
from Pufs.FunctionalPuf import *
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py <attack>")
        print("Choose 'xor' for XOR PUF attack or 'ipuf' for iPUF attack.")
        return

    attack_type = sys.argv[1].lower()

    if attack_type == 'xor':
        res = xor_attk()
        print(res)
    elif attack_type == 'ipuf':
        res = ipuf_attk()
        print(res)
    else:
        print("Invalid attack type. Choose 'xor' or 'ipuf'.")

if __name__ == "__main__":
    main()