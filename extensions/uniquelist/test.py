# -*- coding: utf-8 -*-

"""Description of this file"""

import numpy as np
import uniquelist


def main():
    """Run the main routine of this script"""
    lst = uniquelist.UniqueList()
    print(lst.size())
    x = lst.push_back(2)
    print(x)
    x = lst.push_back(1)
    print(x)
    x = lst.push_back(2)
    print(x)
    x = lst.push_back(3)
    print(x)
    x = lst.push_back(5)
    print(lst.size())
    print(" --- ")
    print(lst.index(2))
    print(lst.index(3))
    print(lst.index(4))
    print(" --- ")
    lst.display()
    # lst.erase_nonzero([0, 1, 0, 1])
    flag = np.array([0, -1, 0, -1])
    lst.erase_nonzero(flag.astype(float))
    lst.display()


if __name__ == "__main__":
    main()
